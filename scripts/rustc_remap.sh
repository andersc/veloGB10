#!/bin/sh
# Cargo rustc wrapper (see .cargo/config.toml): adds one --remap-path-prefix to every rustc call so
# that the build host's absolute directories do not end up in panic locations and other path strings
# of the binary. It changes nothing else: the same rustc, the same arguments, plus that one flag.
# No environment variable is read; the prefix comes from the directory cargo runs rustc in.
#   - a crates.io dependency runs in <registry>/src/<index>/<crate>-<version>; every path under
#     <registry>/src/<index> becomes /cargo/registry/..., e.g. /cargo/registry/tokio-1.52.3/src/...
#   - a crate of this workspace runs in the checkout root; <root>/ becomes empty, so src/x.rs is
#     written src/x.rs whatever the checkout's location.
rustc=$1
shift
cwd=$(pwd -P)
case "$cwd" in
  */registry/src/*/*)
    exec "$rustc" "--remap-path-prefix=${cwd%/*}=/cargo/registry" "$@"
    ;;
  *)
    exec "$rustc" "--remap-path-prefix=$cwd/=" "$@"
    ;;
esac
