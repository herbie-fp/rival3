PKG_NAME := rival3-racket
FFI_MANIFEST := rival3-ffi/Cargo.toml

PLATFORM := $(shell racket -e "(display (path->string (system-library-subpath \#f)))" | tr '\\\\' '/')
LIB_NAME := $(shell racket -e "(display (string-append (if (eq? (system-type) 'windows) \"rival3_ffi\" \"librival3_ffi\") (bytes->string/utf-8 (system-type 'so-suffix))))")

FFI_BUILD := rival3-ffi/target/release/$(LIB_NAME)
PKG_NATIVE_DIR := $(PKG_NAME)/private/native/$(PLATFORM)
PKG_NATIVE_LIB := $(PKG_NATIVE_DIR)/$(LIB_NAME)

.PHONY: build package install update uninstall clean nightly

build:
	cargo build --release --manifest-path=$(FFI_MANIFEST)
	mkdir -p $(PKG_NATIVE_DIR)
	cp $(FFI_BUILD) $(PKG_NATIVE_LIB)

package: build
	raco pkg create --format zip $(PKG_NAME)

install: build
	raco pkg install --user --batch --auto -D --type dir --link --skip-installed $(PKG_NAME)

update: build
	raco pkg update --user --batch --auto -D --type dir --link --skip-uninstalled $(PKG_NAME)

uninstall:
	raco pkg remove --user --batch --auto --force --no-docs $(PKG_NAME)

clean:
	cargo clean
	cargo clean --manifest-path=$(FFI_MANIFEST)
	rm -rf $(PKG_NAME)/private/native
	rm -f $(PKG_NAME).zip $(PKG_NAME).zip.CHECKSUM

nightly:
	bash infra/nightly.sh
