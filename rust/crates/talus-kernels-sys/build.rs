use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let header_rel = "../../../include/talus_kernels.h";
    let header_path = manifest_dir.join(header_rel);

    println!("cargo:rerun-if-changed={header_rel}");
    println!("cargo:rerun-if-env-changed=TALUS_KERNELS_PREBUILT");

    let header_text = fs::read_to_string(&header_path).unwrap_or_else(|e| {
        panic!(
            "failed to read C ABI header at {}: {e}",
            header_path.display()
        )
    });

    let abi_version = parse_abi_version(&header_text).unwrap_or_else(|| {
        panic!(
            "TALUS_KERNELS_ABI_VERSION macro not found in {}",
            header_path.display()
        )
    });
    assert_eq!(
        abi_version, 0,
        "talus-kernels-sys requires TALUS_KERNELS_ABI_VERSION == 0, found {abi_version}"
    );

    let link_kernels = env::var_os("CARGO_FEATURE_LINK_KERNELS").is_some();

    if let Some(prebuilt_os) = env::var_os("TALUS_KERNELS_PREBUILT") {
        let prebuilt_dir = PathBuf::from(prebuilt_os);
        let so_path = prebuilt_dir.join("libtalus_kernels.so");
        if so_path.exists() {
            println!("cargo:rustc-link-search=native={}", prebuilt_dir.display());
            if link_kernels {
                println!("cargo:rustc-link-lib=dylib=talus_kernels");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", prebuilt_dir.display());
            }
            return;
        }
    }

    if link_kernels {
        println!("cargo:rustc-link-lib=dylib=talus_kernels");
    }
}

fn parse_abi_version(header: &str) -> Option<u32> {
    for line in header.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#define") {
            let mut parts = rest.split_whitespace();
            if parts.next() == Some("TALUS_KERNELS_ABI_VERSION") {
                if let Some(val_str) = parts.next() {
                    return val_str.parse::<u32>().ok();
                }
            }
        }
    }
    None
}
