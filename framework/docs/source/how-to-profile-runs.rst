:og:description: Enable run profiling in Flower, inspect summaries via CLI, stream live metrics, and visualize profiling data in the profiling UI.
.. meta::
    :description: Enable run profiling in Flower, inspect summaries via CLI, stream live metrics, and visualize profiling data in the profiling UI.

################
 Profile a Run
################

This guide shows how to enable profiling for a Flower run, inspect results from the
CLI, and view live updates in the profiling UI.

***************
 Prerequisites
***************

- A Flower project with a ``pyproject.toml``.
- A running deployment (for example ``local-deployment``).

**************
 Enable Profiling
**************

You can enable profiling from ``pyproject.toml``:

.. code-block:: toml

    [tool.flwr.app.config]
    profile.enabled = true

Or override it at runtime:

.. code-block:: shell

    flwr run . local-deployment --run-config "profile.enabled=true"

The alias ``profiling.enabled=true`` is also accepted and mapped to
``profile.enabled``.

*****************
 Run and Inspect
*****************

Start a run:

.. code-block:: shell

    flwr run . local-deployment --run-config "profile.enabled=true"

After the run starts, retrieve profile summaries:

.. code-block:: shell

    flwr profile <RUN_ID> . local-deployment

Example:

.. code-block:: shell

    flwr profile 11499293074092490504 . local-deployment

Example output:

.. code-block:: text

    Run Profile Summary
    Task          Scope   Round  Node   Avg (ms)  Max (ms)  Avg Mem (MB)  Avg ΔMem (MB)  Avg Read (MB)  Avg Write (MB)  Disk Src  Count
    aggregate     server  1      server      1.62      1.62        812.34           0.12           0.00            0.00  process       1
    train         client  1      2419     5031.32   7935.37       2048.11          85.42           1.25            0.88  process       4
    evaluate      client  1      2419      512.41    644.22       2050.02           1.91           0.03            0.01  process       4

    Network Profile
    Task        Scope   Round  Node   Avg (ms)  Max (ms)  Count
    downstream  server  1      server  12034.29  15050.85      4
    upstream    server  1      server      5.94      7.57      4
    combined    server  1      server  12040.24  15055.80      4

Values vary by workload and environment.

To stream live updates while the run is active:

.. code-block:: shell

    flwr profile <RUN_ID> . local-deployment --live

Use JSON output if needed:

.. code-block:: shell

    flwr profile <RUN_ID> . local-deployment --format json

*********************
 What Gets Collected
*********************

Flower records:

- Server tasks (for example ``aggregate``, ``network_upstream``,
  ``network_downstream``, ``send_and_receive``).
- Client tasks (for example ``train``, ``evaluate``, ``query`` and task actions).
- Timing (ms) and memory statistics (MB).
- Disk read/write deltas (MB), when supported by the environment.

Disk metrics use a fallback chain for better portability:

1. Per-process I/O counters (preferred).
2. System-wide I/O counters (fallback).
3. ``resource.getrusage`` block counters (final fallback).

The summary includes ``disk_source`` (``process``, ``system``, ``resource``, or
``mixed``).

*************************
 Launch the Profiling UI
*************************

The profiling UI is provided in ``examples/profiling-ui`` and queries the Control API
directly.

From the repository root:

.. code-block:: shell

    pip install flask
    python examples/profiling-ui/app.py \
      --app examples/quickstart-pytorch \
      --federation local-deployment \
      --address 127.0.0.1:9093 \
      --insecure

Then open:

.. code-block:: text

    http://127.0.0.1:5000

Enter the run ID to inspect summary and time-series views.
