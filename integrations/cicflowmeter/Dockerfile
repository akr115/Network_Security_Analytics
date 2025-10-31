FROM python:3.11

WORKDIR /app

# Install only what's necessary for Scapy to read PCAPs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy your repo
COPY . /app

# Upgrade pip and install uv
RUN pip install --upgrade pip
RUN pip install uv

# Run uv sync to setup venv
RUN uv sync

# CMD: process all PCAPs from /pcaps and write flows to /flows
CMD ["/bin/bash", "-c", "\
source .venv/bin/activate && \
for f in /pcaps/*.pcap; do \
  echo 'Processing $f'; \
  cicflowmeter --file \"$f\" -c /flows/$(basename \"$f\" .pcap)_flows.csv; \
done"]
