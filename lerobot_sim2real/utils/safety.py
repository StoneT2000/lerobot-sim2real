### Safety setups. Close environments/turn off robot upon ctrl+c ###
import sys
import signal

def setup_safe_exit(sim_env = None, real_env = None, real_agent = None):
    def signal_handler(sig = None, frame = None):
        print("\nCtrl+C detected. Exiting gracefully...")
        try:
            if real_agent is not None:
                real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
        except Exception:
            pass
        try:
            if real_env is not None:
                real_env.close()
        except Exception:
            pass
        try:
            sim_env.close()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    import atexit
    def exit_handler():
        print("\nScript finished. Exiting gracefully...")
        try:
            if real_agent is not None:
                real_agent.reset(sim_env.unwrapped.agent.keyframes["rest"].qpos)
        except Exception:
            pass
        try:
            if real_env is not None:
                real_env.close()
        except Exception:
            pass
        try:
            sim_env.close()
        except Exception:
            pass
    atexit.register(exit_handler)