Exception in thread Thread-1 (_listen_clear_key):
Traceback (most recent call last):
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/home/qwe/wokonsmall/lerobot_diy/src/lerobot/scripts/server/robot_client.py", line 157, in _listen_clear_key
    keyboard.wait('right')
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/__init__.py", line 881, in wait
    remove = add_hotkey(hotkey, lambda: lock.set(), suppress=suppress, trigger_on_release=trigger_on_release)
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/__init__.py", line 639, in add_hotkey
    _listener.start_if_necessary()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/_generic.py", line 35, in start_if_necessary
    self.init()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/__init__.py", line 196, in init
    _os_keyboard.init()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/_nixkeyboard.py", line 113, in init
    build_device()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/_nixkeyboard.py", line 109, in build_device
    ensure_root()
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/keyboard/_nixcommon.py", line 174, in ensure_root
    raise ImportError('You must be root to use this library on linux.')
ImportError: You must be root to use this library on linux.
INFO 2025-08-12 21:32:15 t_client.py:187 Sending policy instructions to policy server
^CTraceback (most recent call last):
  File "/home/qwe/wokonsmall/lerobot_diy/src/lerobot/scripts/server/robot_client.py", line 614, in <module>
    async_client()  # run the client
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/draccus/argparsing.py", line 225, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
  File "/home/qwe/wokonsmall/lerobot_diy/src/lerobot/scripts/server/robot_client.py", line 592, in async_client
    if client.start():
  File "/home/qwe/wokonsmall/lerobot_diy/src/lerobot/scripts/server/robot_client.py", line 194, in start
    self.stub.SendPolicyInstructions(policy_setup)
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/grpc/_channel.py", line 1175, in __call__
    state, call = self._blocking(
  File "/home/qwe/anaconda3/envs/lerobot/lib/python3.10/site-packages/grpc/_channel.py", line 1162, in _blocking
    event = call.next_event()
  File "src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi", line 388, in grpc._cython.cygrpc.SegregatedCall.next_event
  File "src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi", line 211, in grpc._cython.cygrpc._next_call_event
  File "src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi", line 205, in grpc._cython.cygrpc._next_call_event
  File "src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi", line 97, in grpc._cython.cygrpc._latent_event
  File "src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi", line 80, in grpc._cython.cygrpc._internal_latent_event
  File "src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi", line 61, in grpc._cython.cygrpc._next