FROM pklehre/niso2020-lab2-msc

ADD emc918.py /bin

RUN apt-get update
RUN apt-get -y install python3

# Verbose Commands
#CMD ["-username", "emc918", "-submission", "python3 /bin/emc918.py", "-verbose", "-questions", "1"]
#CMD ["-username", "emc918", "-submission", "python3 /bin/emc918.py", "-verbose", "-questions", "2"]
#CMD ["-username", "emc918", "-submission", "python3 /bin/emc918.py", "-verbose", "-questions", "3"]

CMD ["-username", "emc918", "-submission", "python3 /bin/emc918.py"]
