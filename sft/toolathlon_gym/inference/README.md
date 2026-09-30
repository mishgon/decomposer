# Private lmrouter Tunnel

Use this when the evaluation host cannot reach lmrouter but an OCC host can:

```text
Host/container HTTPS client -> private Unix socket -> Hertz loopback :18443
  -> SSH reverse tunnel -> OCC -> lmrouter.2a2i.org:443
```

The client keeps `https://lmrouter.2a2i.org/v1` as its URL and verifies the
server's TLS certificate. The relay forwards encrypted bytes; it has no API key.
The tunnel and relay run independently of collection and survive disconnecting
the laptop. Restart their tmux sessions after a host reboot.

## Restricted SSH Key

On OCC, configure a reachable host alias in `~/.ssh/config` (Hertz currently
uses port 44444). Replace the host and user with your connection details:

```text
Host hertz-lmrouter
    HostName HERTZ_ADDRESS
    User YOUR_USER
    Port 44444
```

Generate a dedicated key and verify the Hertz SSH host key:

```bash
ssh-keygen -t ed25519 -f "$HOME/.ssh/lmrouter_hertz" -N '' -C lmrouter-tunnel
ssh hertz-lmrouter true
```

Copy the contents of `~/.ssh/lmrouter_hertz.pub` to Hertz's
`~/.ssh/authorized_keys`, prefixed with these options on the same line:

```text
restrict,port-forwarding,permitlisten="127.0.0.1:18443",permitopen="reserved.invalid:1",command="/bin/false" ssh-ed25519 PUBLIC_KEY lmrouter-tunnel
```

Keep the private key on OCC. These options disable shell access, PTY, agent/X11
forwarding and usable local forwarding. They permit only the specified reverse
listener. SSH restricts the listener; the client script selects its destination.

## Tunnel on OCC

From a checkout on OCC, use that host alias:

```bash
tmux -L lmrouter new-session -d -s hertz-router \
  "bash '$PWD/sft/toolathlon_gym/inference/router_tunnel.sh' hertz-lmrouter '$HOME/.ssh/lmrouter_hertz'"
tmux -L lmrouter attach -t hertz-router
```

Detach with Ctrl-b d. Stop with Ctrl-c. The launcher retries failed SSH sessions
with a delay that increases from 2 to 30 seconds and uses SSH keepalives.
The tmux socket name after `-L` must match when starting and attaching.

On Hertz, verify the TLS route before using credentials (HTTP 401 is expected):

```bash
curl --connect-to lmrouter.2a2i.org:443:127.0.0.1:18443 \
  https://lmrouter.2a2i.org/v1/models
```

## Relay and Docker on Hertz

From the repository root:

```bash
tmux new-session -d -s lmrouter-relay \
  "$PWD/.venv/bin/python -m sft.toolathlon_gym.inference.socket_relay --socket '$HOME/.local/share/lmrouter-relay/router.sock'"
```

Load credentials and select the socket before starting the model registry:

```bash
set -a
source "$HOME/.local/share/environment/lmrouter.env"
set +a
export LLM_PROXY_UNIX_SOCKET="$HOME/.local/share/lmrouter-relay/router.sock"
```

That environment file needs `LLM_PROXY_MASTER_KEY`. The registry defines the URL
and generation parameters directly. The runner mounts only the socket directory
into Docker and adjusts its socket path to `/run/model-proxy/router.sock`.
It passes the API key by environment variable name, never as a literal CLI value.

Keep the socket directory private (0700). A clean relay shutdown removes its
socket. If an unclean exit leaves a stale socket, verify no relay owns it before
removing it and restarting. For an already-running tunnel, reuse its listener
and relay rather than starting a second session on the same port.

## Refresh the Runtime Image

`Dockerfile.refresh` copies the current core and Gym onto a pinned, validated
runtime image without downloading dependencies. Pass its ID as `RUNTIME_IMAGE`;
record both the base and resulting image IDs with the run.
