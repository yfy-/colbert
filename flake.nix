{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        condaFile = if pkgs.stdenv.isDarwin then "conda_env_cpu.yml" else "conda_env.yml";
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            micromamba
            git
          ];

          shellHook = ''
            export FLAKE_ROOT="$(git rev-parse --show-toplevel)"
            export MAMBA_ROOT_PREFIX="$FLAKE_ROOT/.mamba"
            if [ ! -d "$MAMBA_ROOT_PREFIX/envs/colbert" ]; then
              micromamba create -f ${condaFile} -n colbert -y
              pip install python-lsp-server python-lsp-black python-lsp-isort
            fi

            export PATH=$MAMBA_ROOT_PREFIX/envs/colbert/bin:$PATH
            export CONDA_PREFIX=$MAMBA_ROOT_PREFIX/envs/colbert
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib
          '';
        };
      }
    );
}
