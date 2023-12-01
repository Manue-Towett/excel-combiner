import os
import re
import sys
import dataclasses
import configparser
from datetime import date
from typing import Optional

import pandas as pd

from utils import Logger

config = configparser.ConfigParser()

with open("./settings/settings.ini", "r") as file:
    config.read_file(file)

OUTPUT_PATH = config.get("paths", "output")

INPUT_PATH = config.get("paths", "input")

COMBINED_LAST_INDEX = int(config.get("csv_count", "max_rows"))

COLUMNS = ["Title", "Url", "Image", "Price"]

@dataclasses.dataclass
class FileStats:
    """Store file descriptions i.e. file name, products count and error"""
    file_name: str
    products_count: int
    error: Optional[str] = None

@dataclasses.dataclass
class SameDomainFiles:
    """Stores old files from the same domain"""
    domain: str
    file_paths: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Columns:
    """stores columns information"""
    title: str
    url: str
    image: str
    price: str

class Combiner:
    """Combines data from several files into one csv file"""
    def __init__(self) -> None:
        self.logger = Logger(__class__.__name__)
        self.logger.info("*****Excel Combiner started*****")

        self.file_stats: list[FileStats] = []
        self.dataframes: list[pd.DataFrame] = []

        self.files = self.__get_files(INPUT_PATH)

    def __get_files(self, path: str) -> Optional[list[str]]:
        files = [f"{path}{file}" for file in os.listdir(path) 
                 if re.search(r"(csv|xlsx)$", file, re.I)]
        
        if len(files):
            info_msg = "files" if len(files) > 1 else "file"

            self.logger.info(f"{len(files)} input {info_msg} found")
            return files
        
        self.logger.error("No excel/csv files found in input directory!")

    @staticmethod
    def __get_columns(columns: list[str], file_stats: FileStats) -> Optional[Columns]: 
        for column in columns:
            if re.search(r"title", column, re.I):
                title = column
            elif re.search(r"url", column, re.I):
                url = column
            elif re.search(r"image", column, re.I):
                image = column
            elif re.search(r"price", column, re.I):
                price = column
        
        try:
            return Columns(title=title, url=url, image=image, price=price)
        
        except:
            _, e_value, _ = sys.exc_info()

            file_stats.error = f"missing column! {e_value}"

    @staticmethod
    def __rename_columns(df: pd.DataFrame, columns: Columns) -> pd.DataFrame:
        return df.rename(columns={columns.title: COLUMNS[0],
                                  columns.url: COLUMNS[1],
                                  columns.image: COLUMNS[2],
                                  columns.price: COLUMNS[-1]})
    
    @staticmethod
    def __read_file(file_path: str) -> Optional[pd.DataFrame]:
        if re.search(r".xlsx$", file_path, re.I):
            return pd.read_excel(file_path)
        elif re.search(r".csv$", file_path, re.I):
            return pd.read_csv(file_path)
    
    @staticmethod
    def __remove_whitespace(df: pd.DataFrame) -> None:
        df[COLUMNS[-1]] = df[COLUMNS[-1]].apply(
            lambda value: str(value).strip()
        )

        df[COLUMNS[0]] = df[COLUMNS[0]].apply(
            lambda value: str(value).strip()
        )
    
    @staticmethod
    def __format_price(df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(subset=COLUMNS[-1], inplace=True)

        df = df.astype({COLUMNS[-1]: str})

        df = df.loc[~df[COLUMNS[-1]].str.contains("-")]

        df = df.loc[df[COLUMNS[-1]].str.contains(r"\$*\d+\.*\d*(?![\-])", regex=True)]

        df.loc[df[COLUMNS[-1]].str.contains(r"\${1,1}\s*", regex=True), 
               COLUMNS[-1]] = df[COLUMNS[-1]].apply(
            lambda value: str(value).strip("$").strip().replace(",", "").replace(" ", "")
        )

        df[COLUMNS[-1]] = df[COLUMNS[-1]].astype(float)

        df[COLUMNS[-1]] =  df[COLUMNS[-1]].map("${:,.2f}".format)

        return df
    
    def __combine_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        price_columns = [c for c in df.columns.values if re.search("price", c, re.I)]

        if len(price_columns) > 1:
            df["Price"] = df[price_columns].min(axis=1)

            [df.drop(columns=c, inplace=True) for c in price_columns if c != "Price"]
        
        return df
    
    def __save_to_csv(self, index: int) -> int: 
        combined_df = pd.concat(self.dataframes).iloc[:COMBINED_LAST_INDEX]

        dates_ = str(date.today()).split("-")

        _date = f"{'.'.join(dates_[-2:])}.{dates_[0][-2:]}"

        combined_name = f'{OUTPUT_PATH}{"combined_results_{}_{}.csv".format(_date, index)}'

        combined_df.to_csv(combined_name, index=False)

        self.logger.info(f"{len(combined_df)} products saved to {combined_name}")

        return len(combined_df)

    def run(self) -> None:
        results_no, index = 0, 1

        for file in self.files:
            name = file.split(INPUT_PATH)[-1]

            self.logger.info("Processing file: {}".format(name))

            df = self.__read_file(file)

            stats = FileStats(file_name=name, products_count=len(df))

            if df is None: continue

            df = self.__combine_price_columns(df)

            columns = self.__get_columns(df.columns.values, stats)

            if columns is None: 
                stats.products_count = None

                self.file_stats.append(stats)

                continue

            df = self.__rename_columns(df, columns)

            self.__remove_whitespace(df)

            df = self.__format_price(df) 

            if len(df):
                self.dataframes.append(df)

                results_no = self.__save_to_csv(index)

            else: 
                self.logger.info("No products from file: {}".format(name))

            self.file_stats.append(stats) 

            if results_no >= COMBINED_LAST_INDEX:
                self.logger.info("Maximum records per combined file reached.")

                index += 1

        file_stats = [dataclasses.asdict(stats) for stats in self.file_stats] 

        pd.DataFrame(file_stats).to_csv("./stats/stats.csv", index=False)

        self.logger.info("Done combining files.")   


if __name__ == "__main__":
    app = Combiner()
    app.run()