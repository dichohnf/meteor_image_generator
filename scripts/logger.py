from datetime import datetime
from pathlib import Path
from typing import Union

class Logger:
  def __init__(self, enable=True):
    self.enable = enable

  def log(self, message: str, type: str = None):
    if self.enable:
      raise NotImplemented("Logging not implemented")

  def disable(self):
    self.enable = False

  def enable(self):
    self.enable = True
    
  def enable_set(self, enable: bool):
    self.enable = enable
  
  def info(self, message: str):
    self.log(message, "INFO")
    
  def warning(self, message: str):
    self.log(message, "WARN")
    
  def error(self, message: str):
    self.log(message, "ERROR")

def now_with_seconds():
  return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class ConsoleLogger(Logger):
  def __init__(self, enable=True):
      super().__init__(enable)
      self.colors = {
        "INFO":  "\033[92m",      # Green
        "WARN":  "\033[93m",      # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
      }

  def log(self, message: str, type: str = None):
    if self.enable:
      # ANSI color codes
      color = self.colors.get(type.upper(), "") if type else ""
      reset = self.colors["RESET"]
      
      if type:
        print(f"{color}[{now_with_seconds()}] [{type.upper()}]{reset} {message}")
      else:
        print(f"{color}[{now_with_seconds()}]{reset} {message}")

class FileLogger(Logger):
  def __init__(self, log_file: Union[Path, str], enable=True):
      super().__init__(enable)
      self.log_file = log_file

  def log(self, message: str, type: str = None):
    if self.enable:
        mode = "a" if Path(self.log_file).exists() else "w"
        with open(self.log_file, mode) as f:
            if type:
                f.write(f"[{now_with_seconds()}] [{type.upper()}] {message}\n")
            else:
                f.write(f"[{now_with_seconds()}] {message}\n")
            
global logger 
logger = ConsoleLogger()