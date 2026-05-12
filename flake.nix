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
              micromamba create -f conda_env.yml -n colbert -y
            fi

            export PATH=$MAMBA_ROOT_PREFIX/envs/colbert/bin:$PATH
            export CONDA_PREFIX=$MAMBA_ROOT_PREFIX/envs/colbert
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib
            pip install python-lsp-server python-lsp-black python-lsp-isort
          '';
        };
      }
    );
}
