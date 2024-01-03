import sinter

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)