#!/usr/bin/env python3
"""Synthetic provider smoke test. Does not open the DB or send email."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.summarizer import Summarizer

if __name__ == "__main__":
    Summarizer().check_providers()
