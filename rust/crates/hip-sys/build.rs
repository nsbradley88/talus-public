use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TALUS_HIP_LIB");
    println!("cargo:rerun-if-env-changed=ROCM_SDK");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    if env::var_os("CARGO_FEATURE_LINK_HIP").is_some() {
        for var in ["ROCM_SDK", "ROCM_PATH"] {
            if let Ok(root) = env::var(var) {
                let lib_dir = PathBuf::from(root).join("lib");
                if lib_dir.is_dir() {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                }
            }
        }

        let opt_rocm = PathBuf::from("/opt/rocm/lib");
        if opt_rocm.is_dir() {
            println!("cargo:rustc-link-search=native={}", opt_rocm.display());
        }

        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
}
