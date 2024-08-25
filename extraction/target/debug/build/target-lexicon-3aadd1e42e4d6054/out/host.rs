
#[allow(unused_imports)]
use crate::Aarch64Architecture::*;
#[allow(unused_imports)]
use crate::ArmArchitecture::*;
#[allow(unused_imports)]
use crate::CustomVendor;
#[allow(unused_imports)]
use crate::Mips32Architecture::*;
#[allow(unused_imports)]
use crate::Mips64Architecture::*;
#[allow(unused_imports)]
use crate::Riscv32Architecture::*;
#[allow(unused_imports)]
use crate::Riscv64Architecture::*;
#[allow(unused_imports)]
use crate::X86_32Architecture::*;

/// The `Triple` of the current host.
pub const HOST: Triple = Triple {
    architecture: Architecture::Aarch64(Aarch64),
    vendor: Vendor::Apple,
    operating_system: OperatingSystem::Darwin,
    environment: Environment::Unknown,
    binary_format: BinaryFormat::Macho,
};

impl Architecture {
    /// Return the architecture for the current host.
    pub const fn host() -> Self {
        Architecture::Aarch64(Aarch64)
    }
}

impl Vendor {
    /// Return the vendor for the current host.
    pub const fn host() -> Self {
        Vendor::Apple
    }
}

impl OperatingSystem {
    /// Return the operating system for the current host.
    pub const fn host() -> Self {
        OperatingSystem::Darwin
    }
}

impl Environment {
    /// Return the environment for the current host.
    pub const fn host() -> Self {
        Environment::Unknown
    }
}

impl BinaryFormat {
    /// Return the binary format for the current host.
    pub const fn host() -> Self {
        BinaryFormat::Macho
    }
}

impl Triple {
    /// Return the triple for the current host.
    pub const fn host() -> Self {
        Self {
            architecture: Architecture::Aarch64(Aarch64),
            vendor: Vendor::Apple,
            operating_system: OperatingSystem::Darwin,
            environment: Environment::Unknown,
            binary_format: BinaryFormat::Macho,
        }
    }
}
