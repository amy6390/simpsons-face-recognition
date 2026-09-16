from pathlib import Path

import typer
import kagglehub 

from config import EXTERNAL_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    output_path: Path = EXTERNAL_DATA_DIR
):
    
    alexattia_the_simpsons_characters_dataset_path = kagglehub.dataset_download('alexattia/the-simpsons-characters-dataset', output_dir = str(output_path), force_download = True)
    print('Data source import complete.')
    print(alexattia_the_simpsons_characters_dataset_path)


if __name__ == "__main__":
    app()
