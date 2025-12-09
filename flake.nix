{
  description = "TicTacToe AI Application with Reinforcement Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        # Development shell with all dependencies
        devShells.default = pkgs.mkShell {
          name = "tictactoe-ai-dev";

          packages = with pkgs; [
            # Python environment
            python311

            # Python packages
            python311Packages.numpy
            python311Packages.tkinter
          ];
        };
      };
    };
}
