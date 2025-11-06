#MDP-specific parameters that differ by dataset
SMALL = dict(
    S=100,
    A=4,
    GAMMA=0.95,
    episodic=False
)

MEDIUM = dict(
    S=50_000,
    A=7,
    GAMMA=1.0,  #undiscounted episodic task
    episodic=True
)

LARGE = dict(
    S=302_020,
    A=9,
    GAMMA=0.95,
    episodic=False
)
