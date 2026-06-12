// otpencrypt/src/main.rs
//
// XOR a file against a one-time pad, with operational-security hardening:
//   * buffered, chunked streaming (flat memory, few syscalls)
//   * pad-length checked up front (no half-written output on failure)
//   * original filename AND real length carried in an encrypted header;
//     ciphertext gets a random, non-revealing name
//   * optional length-hiding padding (--block / --pad-to / --match-pad): the
//     real plaintext length is stored encrypted in the header and used as the
//     logical EOF, so trailing random padding is discarded on decrypt
//   * sensitive buffers zeroized after use
//   * memory locked against swap (best effort)
//   * pad and input shredded after use by default (best effort)
//   * Result-based error handling throughout; no panics on user error

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use rand::rngs::OsRng;
use rand::RngCore;
use zeroize::Zeroize;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

const CHUNK_SIZE: usize = 64 * 1024;
const BODY_LEN_BYTES: usize = 8; // u64 real-length prefix
const NAME_LEN_BYTES: usize = 2; // u16 filename-length prefix
const FIXED_HEADER: usize = BODY_LEN_BYTES + NAME_LEN_BYTES;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encrypt a file with a one-time pad
    Encrypt(Job),
    /// Decrypt a file with a one-time pad
    Decrypt(Job),
}

#[derive(ClapArgs)]
struct Job {
    /// File to process
    #[arg(short, long)]
    input: String,

    /// One-time pad file
    #[arg(short, long)]
    pad: String,

    /// Output path (default: random name on encrypt, recovered name on decrypt)
    #[arg(short, long)]
    output: Option<String>,

    /// Overwrite the output if it already exists
    #[arg(short, long)]
    force: bool,

    /// Length hiding: pad the ciphertext up to a multiple of this many bytes
    #[arg(long)]
    block: Option<u64>,

    /// Length hiding: pad the ciphertext to exactly this many bytes
    #[arg(long = "pad-to")]
    pad_to: Option<u64>,

    /// Length hiding: pad the ciphertext to the size of the pad file
    #[arg(long)]
    match_pad: bool,

    /// Do NOT shred the pad after use (default is best-effort shred)
    #[arg(long)]
    no_pad_shred: bool,

    /// Do NOT shred the input after use (default is best-effort shred)
    #[arg(long)]
    no_input_shred: bool,
}

fn main() -> Result<()> {
    lock_memory();
    let cli = Cli::parse();
    match &cli.command {
        Command::Encrypt(job) => encrypt(job),
        Command::Decrypt(job) => decrypt(job),
    }
}

/// Pin process memory so secrets can't be paged out to swap. Best effort.
#[cfg(unix)]
fn lock_memory() {
    let rc = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if rc != 0 {
        eprintln!(
            "warning: could not lock memory (mlockall failed); secrets may be \
             swapped to disk. Raise RLIMIT_MEMLOCK, run privileged, or disable swap."
        );
    }
}
#[cfg(not(unix))]
fn lock_memory() {}

fn encrypt(job: &Job) -> Result<()> {
    let body_len = fs::metadata(&job.input)
        .with_context(|| format!("reading input {}", job.input))?
        .len();

    let name = Path::new(&job.input)
        .file_name()
        .context("input path has no file name")?
        .to_string_lossy()
        .into_owned();
    let name_bytes = name.as_bytes();
    if name_bytes.len() > u16::MAX as usize {
        bail!("file name is too long to embed");
    }

    let header_len = FIXED_HEADER + name_bytes.len();
    let real_size = header_len as u64 + body_len; // bytes that consume pad

    let pad_size = fs::metadata(&job.pad)
        .with_context(|| format!("reading pad {}", job.pad))?
        .len();
    if pad_size < real_size {
        bail!(
            "pad too short: need {} bytes ({} header + {} data), pad has {}",
            real_size,
            header_len,
            body_len,
            pad_size
        );
    }

    // Padding is free random filler appended after the real ciphertext; it does
    // NOT consume pad. Compute the final on-disk size.
    let target = target_size(real_size, job.block, job.pad_to, job.match_pad, pad_size)?;
    let pad_filler = target - real_size;

    let out_path = job.output.clone().unwrap_or_else(random_name);
    println!(
        "Encrypting {} -> {} ({} data bytes, {} padding bytes, {} total)",
        job.input, out_path, body_len, pad_filler, target
    );

    {
        let mut input = BufReader::new(
            File::open(&job.input).with_context(|| format!("opening {}", job.input))?,
        );
        let mut pad = BufReader::new(
            File::open(&job.pad).with_context(|| format!("opening pad {}", job.pad))?,
        );
        let mut out = BufWriter::new(open_output(&out_path, job.force)?);

        // Encrypted header: [u64 body_len][u16 name_len][name], XORed with pad.
        let mut header = Vec::with_capacity(header_len);
        header.extend_from_slice(&body_len.to_le_bytes());
        header.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        header.extend_from_slice(name_bytes);
        let mut hk = vec![0u8; header.len()];
        pad.read_exact(&mut hk).context("pad ran out while writing header")?;
        for (h, k) in header.iter_mut().zip(hk.iter()) {
            *h ^= *k;
        }
        out.write_all(&header).context("writing header")?;
        header.zeroize();
        hk.zeroize();

        // Body: exactly body_len bytes, XORed with pad.
        xor_stream(&mut input, &mut pad, &mut out, body_len)?;

        // Padding: random filler, not pad-derived, ignored on decrypt.
        write_padding(&mut out, pad_filler)?;

        out.flush().context("flushing output")?;
        out.get_ref().sync_all().context("syncing output to disk")?;
    }

    shred_after(job, &job.input);
    Ok(())
}

fn decrypt(job: &Job) -> Result<()> {
    let ct_size = fs::metadata(&job.input)
        .with_context(|| format!("reading input {}", job.input))?
        .len();
    let pad_size = fs::metadata(&job.pad)
        .with_context(|| format!("reading pad {}", job.pad))?
        .len();
    if ct_size < FIXED_HEADER as u64 {
        bail!("input is too small to be a valid encrypted file");
    }

    let mut input = BufReader::new(
        File::open(&job.input).with_context(|| format!("opening {}", job.input))?,
    );
    let mut pad = BufReader::new(
        File::open(&job.pad).with_context(|| format!("opening pad {}", job.pad))?,
    );

    // Fixed part of the header: [u64 body_len][u16 name_len].
    let mut hb = [0u8; FIXED_HEADER];
    input.read_exact(&mut hb).context("reading header")?;
    let mut hk = [0u8; FIXED_HEADER];
    pad.read_exact(&mut hk).context("pad ran out reading header")?;
    for (b, k) in hb.iter_mut().zip(hk.iter()) {
        *b ^= *k;
    }
    let body_len = u64::from_le_bytes(hb[..BODY_LEN_BYTES].try_into().unwrap());
    let name_len = u16::from_le_bytes(hb[BODY_LEN_BYTES..].try_into().unwrap()) as usize;

    let header_len = FIXED_HEADER + name_len;
    // The real (pad-consuming) region must fit inside both the ciphertext and pad.
    if header_len as u64 + body_len > ct_size {
        bail!("corrupt file or wrong pad: header/length doesn't fit the file");
    }
    if pad_size < header_len as u64 + body_len {
        bail!(
            "pad too short: need {} bytes, pad has {}",
            header_len as u64 + body_len,
            pad_size
        );
    }

    // Variable part of the header: the name.
    let mut name_buf = vec![0u8; name_len];
    input.read_exact(&mut name_buf).context("reading embedded name")?;
    let mut nk = vec![0u8; name_len];
    pad.read_exact(&mut nk).context("pad ran out reading name")?;
    for (b, k) in name_buf.iter_mut().zip(nk.iter()) {
        *b ^= *k;
    }
    let recovered = String::from_utf8_lossy(&name_buf).into_owned();
    name_buf.zeroize();
    nk.zeroize();

    // Defensive: only ever use the basename, to prevent path traversal.
    let safe_name = Path::new(&recovered)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "decrypted.out".to_string());
    let out_path = job.output.clone().unwrap_or(safe_name);
    println!("Decrypting {} -> {} ({} bytes)", job.input, out_path, body_len);

    {
        let mut out = BufWriter::new(open_output(&out_path, job.force)?);
        // Only body_len bytes are real; anything after is padding we ignore.
        xor_stream(&mut input, &mut pad, &mut out, body_len)?;
        out.flush().context("flushing output")?;
        out.get_ref().sync_all().context("syncing output to disk")?;
    }
    drop(input);
    drop(pad);

    shred_after(job, &job.input);
    Ok(())
}

/// XOR exactly `remaining` bytes of `input` against `pad`, writing to `out`.
fn xor_stream<R: Read, P: Read, W: Write>(
    input: &mut R,
    pad: &mut P,
    out: &mut W,
    mut remaining: u64,
) -> Result<()> {
    let mut ibuf = vec![0u8; CHUNK_SIZE];
    let mut pbuf = vec![0u8; CHUNK_SIZE];
    while remaining > 0 {
        let want = (CHUNK_SIZE as u64).min(remaining) as usize;
        input
            .read_exact(&mut ibuf[..want])
            .context("input shorter than expected")?;
        pad.read_exact(&mut pbuf[..want])
            .context("pad ran out (shorter than the data)")?;
        for k in 0..want {
            ibuf[k] ^= pbuf[k];
        }
        out.write_all(&ibuf[..want]).context("writing output")?;
        remaining -= want as u64;
    }
    ibuf.zeroize();
    pbuf.zeroize();
    Ok(())
}

/// Append `n` bytes of random, non-pad-derived padding.
fn write_padding<W: Write>(out: &mut W, mut n: u64) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let mut rng = OsRng;
    let mut buf = vec![0u8; CHUNK_SIZE];
    while n > 0 {
        let want = (CHUNK_SIZE as u64).min(n) as usize;
        rng.fill_bytes(&mut buf[..want]);
        out.write_all(&buf[..want]).context("writing padding")?;
        n -= want as u64;
    }
    buf.zeroize();
    Ok(())
}

/// Decide the final on-disk size given the length-hiding flags.
fn target_size(
    real: u64,
    block: Option<u64>,
    pad_to: Option<u64>,
    match_pad: bool,
    pad_size: u64,
) -> Result<u64> {
    let chosen = block.is_some() as u8 + pad_to.is_some() as u8 + match_pad as u8;
    if chosen > 1 {
        bail!("use at most one of --block, --pad-to, --match-pad");
    }
    if let Some(b) = block {
        if b == 0 {
            bail!("--block must be greater than 0");
        }
        return Ok(((real + b - 1) / b) * b);
    }
    if let Some(f) = pad_to {
        if f < real {
            bail!("--pad-to {} is smaller than the {} bytes this file needs", f, real);
        }
        return Ok(f);
    }
    if match_pad {
        // pad_size >= real is already guaranteed by the caller's length check.
        return Ok(pad_size);
    }
    Ok(real) // no padding
}

/// Open the output file: 0600 on Unix, refuse to clobber unless --force.
fn open_output(path: &str, force: bool) -> Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true);
    #[cfg(unix)]
    opts.mode(0o600);
    if force {
        opts.create(true).truncate(true);
    } else {
        opts.create_new(true);
    }
    opts.open(path).with_context(|| {
        if force {
            format!("could not create output file {}", path)
        } else {
            format!("output file {} already exists (pass --force to overwrite)", path)
        }
    })
}

/// Best-effort shredding after a successful operation.
fn shred_after(job: &Job, input_path: &str) {
    if !job.no_input_shred {
        match shred_file(input_path) {
            Ok(()) => println!("Shredded input {}", input_path),
            Err(e) => eprintln!("warning: could not shred input {}: {}", input_path, e),
        }
    }
    if !job.no_pad_shred {
        match shred_file(&job.pad) {
            Ok(()) => println!("Shredded pad {}", job.pad),
            Err(e) => eprintln!("warning: could not shred pad {}: {}", job.pad, e),
        }
    }
    println!("Done.");
}

/// Best-effort secure delete: overwrite with random bytes, fsync, then unlink.
///
/// NOTE: genuinely best-effort. On SSDs (wear leveling) and CoW/journaling
/// filesystems, overwrite-in-place often does NOT touch the original blocks.
/// For real assurance, keep pads on single-use removable media and physically
/// destroy it.
fn shred_file(path: &str) -> Result<()> {
    let len = fs::metadata(path)?.len();
    {
        let mut f = OpenOptions::new()
            .write(true)
            .open(path)
            .with_context(|| format!("opening {} for shredding", path))?;
        let mut rng = OsRng;
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut remaining = len;
        while remaining > 0 {
            let n = (CHUNK_SIZE as u64).min(remaining) as usize;
            rng.fill_bytes(&mut buf[..n]);
            f.write_all(&buf[..n])?;
            remaining -= n as u64;
        }
        f.flush()?;
        f.sync_all()?;
        buf.zeroize();
    }
    fs::remove_file(path)?;
    Ok(())
}

/// A random, non-revealing output name (24 hex chars, no extension).
fn random_name() -> String {
    let mut b = [0u8; 12];
    let mut rng = OsRng;
    rng.fill_bytes(&mut b);
    let mut s = String::with_capacity(24);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}
