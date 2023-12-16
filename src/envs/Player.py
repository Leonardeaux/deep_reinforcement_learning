#definitions of constants
MONK_TO_INT = {"RED": 1, "GREEN": 2}
CHECKPOINT_TO_INT = {"RED": 3, "GREEN": 4}
MAX_MONK = 3
DOUBLE_CHECKPOINT = 5
MONK_WITH_OPPONENT_CHECKPOINT = 6


class Player():
    def __init__(self, color: str, random_player: bool, player_type: str) -> None:
        self.piece = color
        self.piece_int = MONK_TO_INT[color]
        self.random_player = random_player
        self.checkpoint_int = CHECKPOINT_TO_INT[color]
        self.nb_piece = MAX_MONK
        self.player_type = player_type
        self.double_checkpoint = DOUBLE_CHECKPOINT
        self.piece_with_opponent_checkpoint = MONK_WITH_OPPONENT_CHECKPOINT
        self.piece_won = 0

    def reload_piece(self, nb_monk_won: int) -> None:
        self.nb_piece = MAX_MONK - nb_monk_won