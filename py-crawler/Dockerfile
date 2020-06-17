FROM ubuntu:16.04

# Installing Java
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# --- adding the openssh-server
RUN apt-get clean && apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y openssh-server supervisor
RUN mkdir /var/run/sshd
RUN mkdir /supervisor-logs

RUN adduser notroot && adduser notroot users

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 22
# --------

RUN echo 'root:root' | chpasswd
RUN echo 'notroot:123456' | chpasswd

RUN mkdir updates
RUN mkdir snapshots

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN cd /

ENV CONFIG_MODE=fixed
ENV PRODUCTS=BTC-USD,ETH-USD

COPY snapshot-crawler.py /
COPY update-crawler.py /
COPY utils.py /
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN pip3 install cbpro
RUN pip3 install schedule
RUN pip3 install orjson

RUN chmod -R 777 updates
RUN chmod -R 777 snapshots

VOLUME ["/updates"]

CMD /usr/bin/supervisord