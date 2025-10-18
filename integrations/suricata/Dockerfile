FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:oisf/suricata-stable && \
    apt-get update && \
    apt-get install -y suricata jq python3 python3-pip curl && \
    suricata-update && \
    mkdir -p /etc/suricata/rules && \
    cp /var/lib/suricata/rules/suricata.rules /etc/suricata/rules/suricata.rules && \
    apt-get clean && \
    mkdir -p /var/log/suricata && chmod 755 /var/log/suricata

COPY suricata.yaml /etc/suricata/suricata.yaml

# # Use this script to generate some network traffic in the container
# COPY generate_network_traffic.sh generate_network_traffic.sh

# COPY http_small_file.pcap http_small_file.pcap
# COPY tcp_file_downloads.pcap tcp_file_downloads.pcap
# COPY pipekat-poshc2.pcap pipekat-poshc2.pcap
COPY example.pcap example.pcap

VOLUME ["/var/log/suricata"]

CMD ["suricata", "-c", "/etc/suricata/suricata.yaml", "-r", "example.pcap", "-l", "/var/log/suricata"]