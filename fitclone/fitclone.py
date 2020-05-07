#!/usr/bin/env python3
import os
import yaml
import sys
import time

exec(open('revive_conditional.py').read())
CondExp().run_with_config_file('../data/SA501/sa501_config.yaml')
