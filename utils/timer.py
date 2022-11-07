from time import time


start_time = 0


def start():
    global start_time
    start_time = int(time())


def end():
    end_time = int(time())
    duration = end_time - start_time
    return (duration, time_parser(duration))


def print_duration(action):
    print(f"Used time for {action}: {end()[1]}")


def time_parser(time):
    time = int(time)
    hours = format(int(time / 3600), "02d")
    minutes = format(int((time % 3600) / 60), "02d")
    secs = format(int(time % 3600 % 60), "02d")
    return f"{hours}:{minutes}:{secs}"
