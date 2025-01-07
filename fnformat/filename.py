"""
FilenameFormat class
"""

import os
import warnings
from copy import deepcopy, copy
from pathlib import Path
from typing import List, Union, Dict, Optional, Any

import pandas as pd

from tiuseful.string import StringFormat


class FilenameFormat(StringFormat):
    """
    A FilenameFormat represents a filename format that includes embedded metadata.
    For example "irr_{label}_{date}_{model}~{serial:.3f}.csv".
    The class offers methods for matching file names, filtering
    directories and changing data to and from filename string representation.

    Python f-string formatting is used to convert metadata into strings. In order to
    parse metadata from a filename or match a filename with the format specifier
    the given format specifier is converted to a regex expression. In order to be
    able to uniquely extract metadata from the filename the filename format should be
    chosen so that metadata fields are separated by characters that will not be used
    in the fields themselves, otherwise parsing the string would be ambiguous.
    Separator chars (e.g. underscores) can be defined on class initialisation.

    "ext" is a special field if it comes at the end of the format specifier. The regex
    for this will specifically look for a .extension type format so the ext field can
    be next to another field with no separator e.g.
    "irr_{label}_{date}_{model}~{serial:.3f}{ext}"


    Methods:
        to_str: Convert a dictionary of field values to a string
        to_dict: Convert a string matching the format specifier into a dictionary
        is_match: Check if a string is a match for the format specifier
        filt_dir: Filter directory for files matching the pattern. Extra args are used
            to filter for certain types of metadta. Return MetaFN objects.
        filt_dir_path: As above but return list of paths
        filt_dir_pandas: As above but return a pandas DataFrame with file paths and
            metadata

    Attributes:
        regex: the regex expression derived from the format specifier used to
            extract the metadata from the filename and do matching. You can custom
            modify this from what's generated
        name: Just a convenience attribute for the instance
        description: Just a convenience attribute for the instance
        reader: function that can open the file. e.g. pd.read_csv
    """

    # special field for file extension
    file_ext_field = "ext"
    # regex for a file extension (. followed by 1 or more alphanumeric characters)
    ext_field_regex = r"\.[a-zA-Z0-9]+$"

    def __init__(
        self,
        fmt: str,
        date_fmt: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        attrs: Optional[dict] = None,
        reader: Optional[Any] = None,
        sep: Optional[Union[str, List[str]]] = None,
        field_regex: Optional[Dict[str, str]] = None,
    ):
        """
        Initialise a FilenameFormat instance with a format specifier

        :param fmt: The format specifier with fields for metadata
            e.g. "dosimeter_{loc}_{date}_{v1}~{v2}.csv"
        :param date_fmt: dict of date formatting strings for any datetime like fields
            in the fmt specifier. e.g. {"date": "%Y-%m-%d}
        :param name: Optional parameter that can be used to set the name of the
            FilenameFormat. Just for convenience.
        :param description: Optional parameter that can be given to describe the
            FilenameFormat. Just for convenience.
        :param attrs: dict of arbitrary metadata associated with this
            filename format (e.g. the name of the file source or a description
            of the typical file contents). These are set as attrs of the class so
            can be accessed through myfilenametype.myattr type notation.
        :param reader: if given should be a function that can take a filepath
            and return an object (normally a function that reads and returns the
            contents of the file)
        :param sep: char or list of characters that separate the format fields (i.e. will
            not be inside any of the data fields). These are helpful to prevent greedy
            searches picking up more fields than they should. Default ["_"].
        :param field_regex: dict with field labels as keys and specific regex strings
            as values. Note if supplying custom regex for a field the sep argument is
            ignored for that field and the user supplied regex is used exactly.
        """
        super().__init__(fmt, date_fmt=date_fmt, sep=sep, field_regex=field_regex)

        bad = [f for f in self.fields if f in MetaFn.protected_names]
        if len(bad) > 0:
            raise ValueError(f"Format fields include protected names {bad}.")
        self.name: str = name
        self.description: str = description
        self._attrs: dict = {} if attrs is None else attrs
        self.attrs = self.attrs
        self.reader = reader

        # add special regex for recognising a file extension
        fr = {} if field_regex is None else copy(field_regex)
        ext_field = self.__class__.file_ext_field
        if self.fields[-1] == ext_field:
            fr[ext_field] = self.__class__.ext_field_regex
            self.regex = self.fmt_to_regex(**fr)

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        for k, v in self.attrs.items():
            if hasattr(self, k):
                warnings.warn(
                    f"Cannot set attr which matches class attr {k}", UserWarning
                )
                continue
            setattr(self, k, v)
        self._attrs = attrs

    def is_match(self, s: Union[str, Path]) -> bool:
        """
        Check if filename matches the format specifier
        """
        return super().is_match(Path(s).name)

    def to_dict(self, fn: Union[str, Path], parse=True) -> dict:
        """
        Convert filename into a dictionary of data

        :param fn: the filename to extract metadata from
        :param parse: whether to attempt to convert str types to appropriate python
            objects. (Converts using date_fmt or attempts to convert using json)

        :return: dictionary with fields matching the fields in the format specifier
            string.
        """
        return super().to_dict(Path(fn).name, parse=parse)

    def to_dicts(
        self, fps: List[Union[str, Path]], parse=True
    ) -> Dict[Union[str, Path], Dict[str, Any]]:
        """
        Convert a list of filenames or paths to a dictionary of metadata

        :param fps: list of filenames or paths to extract metadata from
        :param parse: whether to attempt to convert str types to appropriate python
            objects. (Converts using date_fmt or attempts to convert using json)
        """
        ret = {
            fp: dict(fn=fp.name, fp=fp, fmt=self, **self.to_dict(fp, parse=parse))
            for fp in fps
        }
        return ret

    def to_pandas(self, fps: List[Union[str, Path]], parse=True) -> pd.DataFrame:
        """
        Convert a list of filenames or paths to a pandas DataFrame of metadata

        :param fps: list of filenames or paths to extract metadata from
        :param parse: whether to attempt to convert str types to appropriate python
            objects. (Converts using date_fmt or attempts to convert using json)
        """
        df = pd.DataFrame.from_dict(self.to_dicts(fps, parse=parse), orient="index")
        df = df.reset_index(drop=True)
        return df

    def filt(
        self,
        fps: List[Union[str, Path]],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
    ) -> List[Union[str, Path]]:
        """
        Filter a list of filenames or paths for those matching the fmt and given filters

        :param fps: list of paths to filter
        :param filters: dictionary with filters. Keys are fields in the format
            string, values are the strings to be matched. For example:
            {"field1": "abc", "date_field": Timestamp("2024-01-01"), "field2": 0.1}.
            Class instance date_fmt will be applied to turn date into a string.
        :param ffilters: dictionary of advanced filters. Keys are fields in the format
            string and values are functions that take a string and return a boolean.
            e.g. {"field1": lambda x: x in ["abc", "def"]}.

        :return matches: list of matching filenames or paths
        """
        fns = {Path(fp).name: fp for fp in fps}
        matches = super().filt(list(fns), filters=filters, ffilters=ffilters)
        return [v for k, v in fns.items() if k in matches]

    def filt_dir(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
    ) -> List[Path]:
        """
        Filter a directory for those matching the fmt and given filters

        :param dp: directory to filter
        :param filters: dictionary with filters. Keys are fields in the format
            string, values are the strings to be matched. For example:
            {"field1": "abc", "date_field": Timestamp("2024-01-01"), "field2": 0.1}.
            Class instance date_fmt will be applied to turn date into a string.
        :param ffilters: dictionary of advanced filters. Keys are fields in the format
            string and values are functions that take a string and return a boolean.
            e.g. {"field1": lambda x: x in ["abc", "def"]}.
        :param recursive: whether to search recursively through directories below dp

        :return matches: list of matching paths
        """
        if recursive:
            matches = []
            for root, dns, fns in os.walk(str(dp)):
                matches += self.filt(
                    [Path(root) / fn for fn in fns], filters=filters, ffilters=ffilters
                )
        else:
            root, dns, fns = next(os.walk(str(dp)))
            matches = self.filt(
                [Path(root) / fn for fn in fns], filters=filters, ffilters=ffilters
            )
        return matches

    def filt_dir_dict(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        parse=True,
    ) -> Dict[Union[str, Path], Dict[str, Any]]:
        """
        Filter a directory for those matching the fmt and given filters and return a
        dict

        :param dp: directory to filter
        :param filters: dictionary with filters. Keys are fields in the format
            string, values are the strings to be matched. For example:
            {"field1": "abc", "date_field": Timestamp("2024-01-01"), "field2": 0.1}.
            Class instance date_fmt will be applied to turn date into a string.
        :param ffilters: dictionary of advanced filters. Keys are fields in the format
            string and values are functions that take a string and return a boolean.
            e.g. {"field1": lambda x: x in ["abc", "def"]}.
        :param recursive: whether to search recursively through directories below dp
        :param parse: whether to attempt to convert str types to appropriate python
            objects. (Converts using date_fmt or attempts to convert using json)

        :return matches: dict of matching paths and the corresponding metadata
        """
        matches = self.filt_dir(
            dp, filters=filters, ffilters=ffilters, recursive=recursive
        )
        return self.to_dicts(matches, parse=parse)

    def filt_dir_pandas(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        parse=True,
    ) -> pd.DataFrame:
        """
        Filter a directory for those matching the fmt and given filters and return a
        DataFrame

        :param dp: directory to filter
        :param filters: dictionary with filters. Keys are fields in the format
            string, values are the strings to be matched. For example:
            {"field1": "abc", "date_field": Timestamp("2024-01-01"), "field2": 0.1}.
            Class instance date_fmt will be applied to turn date into a string.
        :param ffilters: dictionary of advanced filters. Keys are fields in the format
            string and values are functions that take a string and return a boolean.
            e.g. {"field1": lambda x: x in ["abc", "def"]}.
        :param recursive: whether to search recursively through directories below dp
        :param parse: whether to attempt to convert str types to appropriate python
            objects. (Converts using date_fmt or attempts to convert using json)

        :return matches: dataframe of matching paths and the corresponding metadata
        """
        d = self.filt_dir_dict(
            dp, filters=filters, ffilters=ffilters, recursive=recursive, parse=parse
        )
        return pd.DataFrame.from_dict(d, orient="index").reset_index(drop=True)

    def read_file(self, fp):
        """
        Load file contents
        """
        if not self.is_match(fp):
            warnings.warn(
                f"Attempting to read file {fp} that does not match the filename "
                f"format {self.fmt}",
                UserWarning,
            )
        return self.reader(fp)


class FilenameFormats:
    """
    Represent a collection of FilenameFormat objects. With match and filter methods
    that behave similarly to the FilenameFormat methods.
    """

    def __init__(self, fmts: List[FilenameFormat]):
        """
        Initialise FilenameFormats instance
        """
        self.fmts: List[FilenameFormat] = fmts

    def is_match(self, s: Union[str, Path]) -> bool:
        """
        Check if filename matches any of the format specifiers
        """
        return any([ff.is_match(s) for ff in self.fmts])

    def which_match(self, s: Union[str, Path]) -> List[FilenameFormat]:
        """
        Return a list of filename formats that match the given path or filename
        """
        return [ff for ff in self.fmts if ff.is_match(s)]

    def filt_dir(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
    ) -> List[Path]:
        __doc__ = FilenameFormat.filt_dir.__doc__
        ret = [
            fp
            for ff in self.fmts
            for fp in ff.filt_dir(
                dp, filters=filters, ffilters=ffilters, recursive=recursive
            )
        ]
        return ret

    def filt_dir_dict(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        parse=True,
    ) -> Dict[Union[str, Path], Dict[str, Any]]:
        __doc__ = FilenameFormat.filt_dir_dict.__doc__
        ret = {}
        for ff in self.fmts:
            ret.update(
                ff.filt_dir_dict(
                    dp,
                    filters=filters,
                    ffilters=ffilters,
                    recursive=recursive,
                    parse=parse,
                )
            )
        return ret

    def filt_dir_pandas(
        self,
        dp: Union[str, Path],
        filters: Optional[Dict[str, Any]] = None,
        ffilters: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        parse=True,
    ) -> pd.DataFrame:
        __doc__ = FilenameFormat.filt_dir_dict.__doc__
        d = self.filt_dir_dict(
            dp, filters=filters, ffilters=ffilters, recursive=recursive, parse=parse
        )
        return pd.DataFrame.from_dict(d, orient="index").reset_index(drop=True)


class MetaFn:
    """
    Holds a filename, its metadata, and an associated FilenameFormat specifier
    that provides methods for converting from metadata to filename and vice
    versa.

    Attributes:
        fn: Filename
        fp: File path
        dp: Directory path
        fmt: FilenameFormat
        fields: names of metadata fields in fn
        meta: dict of metadata extracted from filename
        meta_str: dict of unparsed metadata from filename (no effort to convert
            metadata to python objects)

    Methods:
        to_pandas: Instance data converted to pandas Series object
    """

    # names that should not be in fields
    protected_names = ["fp", "fn", "dp", "format"]

    def __init__(self, fp: Union[str, Path], fmt: FilenameFormat):
        """
        Initialise a Filename instance

        :param fp: File path (can be an absolute path or just a file name)
        :param fmt: FilenameFormat specifier for the filename
        """
        # TODO consider making these private. Behaviour is not guaranteed if they're
        #  changed
        self._fp: Path = Path(fp)
        self._fmt: FilenameFormat = fmt
        self._meta: dict
        self._meta_str: dict
        self._make_metadata()

    def __getattr__(self, n):
        if n in self.fields:
            return self.meta[n]
        raise AttributeError(
            f'"{self.__class__.__name__}" object has no ' f'attribute "{n}"'
        )

    def _make_metadata(self):
        self._meta: dict = self.fmt.to_dict(self.fp)
        self._meta_str: dict = self.fmt.to_dict(self.fp, parse=False)

    @property
    def fp(self) -> Path:
        return self._fp

    @fp.setter
    def fp(self, fp):
        self._fp = Path(fp)
        self._make_metadata()

    @property
    def fmt(self) -> FilenameFormat:
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: FilenameFormat):
        self._fmt = fmt
        self._make_metadata()

    @property
    def fields(self) -> dict:
        return self.fmt.fields

    @property
    def fn(self) -> str:
        """
        File name
        """
        return self.fp.name

    @property
    def dp(self) -> Path:
        """
        Parent directory name
        """
        return self.fp.parent

    @property
    def meta(self) -> Dict[str, Any]:
        """
        Metadata extracted from filename (parsed to python objects where possible)
        """
        return deepcopy(self._meta)

    @property
    def meta_str(self) -> Dict[str, str]:
        """
        Metadata extracted from filename left in string format
        """
        return deepcopy(self._meta_str)

    @property
    def ext(self) -> str:
        return self.fp.suffix

    def to_dict(self, parse=True) -> dict:
        """
        Return file info in a dictionary object
        """
        d = {"fn": self.fn}
        if parse:
            d.update(self.meta)
        else:
            d.update(self.meta_str)
        d.update({"fp": str(self.fp), "format": self.fmt.fmt})
        return d

    def to_pandas(self, parse=True) -> pd.Series:
        """
        Return file info in a pandas Series object.

        :parameter parse: If True then parse data in filename to object types
        """
        return pd.Series(self.to_dict(parse=parse))

    def read(self):
        if self.fmt.reader is not None:
            return self.fmt.reader(str(self.fp))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fp})"
