"""Launch the initial OPD mix through the existing bare-metal OPD lifecycle."""
from opd.wideseek.local import main


if __name__ == '__main__':
    main(training_module='opd.mix.run')
