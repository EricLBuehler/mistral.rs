{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix.url = "github:nix-community/fenix";
    crane = {
      url = "github:ipetkov/crane";
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    { self
    , flake-utils
    , nixpkgs
    , fenix
    , crane
    ,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
          overlays = [
            fenix.overlays.default
            (self: super: {
              rust-dev = super.fenix.stable.withComponents [
                "cargo"
                "clippy"
                "rust-src"
                "rustc"
                "rustfmt"
              ];
            })
          ];
        };
      in
      {
        packages = {
          default =
            let
              craneLib = crane.lib.${system}.overrideToolchain
                fenix.packages.${system}.minimal.toolchain;
            in
            craneLib.buildPackage {
              pname = "mistralrs-server";
              buildInputs = with pkgs; [
                python3
                pkg-config
                openssl
              ];
              checkPhaseCargoCommand = "";
              src = ./.;
            };
          mistralrs-bench =
            let
              craneLib = crane.lib.${system}.overrideToolchain
                fenix.packages.${system}.minimal.toolchain;
            in
            craneLib.buildPackage {
              pname = "mistralrs-bench";
              buildInputs = with pkgs; [
                python3
                pkg-config
                openssl
              ];
              checkPhaseCargoCommand = "";
              src = ./.;
            };
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            rust-dev
            rust-analyzer
            python3
            pkg-config
            openssl
          ];
        };
      }
    );
}
