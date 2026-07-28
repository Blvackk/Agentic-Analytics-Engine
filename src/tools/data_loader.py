#data_loader.py
import pandas as pd

def load_csv(file) -> pd.DataFrame:
    """
    Load a CSV file and return it as a Pandas DataFrame.

    Parameters
    ----------
    file
        A file path or file-like object containing CSV data.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the CSV file is empty or cannot be read.
    """

    try:
        dataframe = pd.read_csv(file)

        if dataframe.empty:
            raise ValueError("The uploaded CSV file is empty.")

        return dataframe

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file contains no data.")

    except pd.errors.ParserError as error:
        raise ValueError(
            f"The CSV file could not be parsed: {error}"
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            "The CSV file uses an unsupported text encoding."
        ) from error

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to load CSV file: {error}"
        ) from error