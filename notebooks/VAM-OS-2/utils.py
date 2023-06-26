import errno
import os


def format_time(time_in_seconds):
    # If the time is less than 60 seconds, return the time formatted as seconds
    if time_in_seconds < 60:
        return f"{time_in_seconds:.2f}s"
    
    # Calculate the number of minutes and seconds
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)
    
    # If the time is less than 1 hour, return the time formatted as minutes and seconds
    if time_in_seconds < 3600:
        return f"{minutes}m {seconds}s"
    
    # Calculate the number of hours, minutes, and seconds
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    
    # Return the time formatted as hours, minutes, and seconds
    return f"{hours}h {minutes}m {seconds}s"


        

def create_dir_if_not_exists(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise