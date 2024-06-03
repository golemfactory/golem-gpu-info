FROM lukemathwalker/cargo-chef:latest as chef
WORKDIR /app

FROM chef AS planner
COPY ./Cargo.toml ./Cargo.lock ./
COPY ./src ./src
COPY ./examples ./examples
RUN cargo chef prepare

FROM chef AS builder
COPY --from=planner /app/recipe.json .
RUN cargo chef cook --release --examples --target x86_64-unknown-linux-gnu
COPY . .
RUN cargo build --release --examples --target x86_64-unknown-linux-gnu
RUN mv ./target/x86_64-unknown-linux-gnu/release/examples/* .

FROM debian:stable-slim AS debian

FROM busybox:stable-glibc AS runtime
WORKDIR /app
COPY --from=debian /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
COPY --from=builder /app/show_gpu_info /usr/local/bin/
