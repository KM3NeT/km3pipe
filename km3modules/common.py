# Filename: common.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""

import sqlite3
from time import time
import json
import importlib

import numpy as np

import km3pipe as kp
from km3pipe import Module, Blob
from km3pipe.tools import prettyln
from km3pipe.sys import peak_memory_usage

from km3pipe.utils.sortprov import _jpp_info_from_file, _read_info_from_prov
from km3pipe.utils.sortprov import *

log = kp.logger.get_logger(__name__)


class Dump(Module):
    """Print the content of the blob.

    Parameters
    ----------
    keys: collection(string), optional [default=None]
        Keys to print. If None, print all keys.
    full: bool, default=False
        Print blob values too, not just the keys?
    """

    def configure(self):
        self.keys = self.get("keys") or None
        self.full = self.get("full") or False
        key = self.get("key") or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        keys = sorted(blob.keys()) if self.keys is None else self.keys
        for key in keys:
            print(key + ":")
            if self.full:
                print(blob[key].__repr__())
            print("")
        print("----------------------------------------\n")
        return blob


class Delete(Module):
    """Remove specific keys from the blob.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to remove.
    """

    def configure(self):
        self.keys = self.get("keys") or set()
        key = self.get("key") or None
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        for key in self.keys:
            blob.pop(key, None)
        return blob


class Keep(Module):
    """Keep only specified keys in the blob.

    Parameters
    ----------
    keys: collection(string), optional
        Keys to keep. Everything else is removed.
    """

    def configure(self):
        self.keys = self.get("keys", default=set())
        key = self.get("key", default=None)
        self.h5locs = self.get("h5locs", default=set())
        if key and not self.keys:
            self.keys = [key]

    def process(self, blob):
        out = Blob()
        for key in blob.keys():
            if key in self.keys:
                out[key] = blob[key]
            elif hasattr(blob[key], "h5loc") and blob[key].h5loc.startswith(
                tuple(self.h5locs)
            ):
                out[key] = blob[key]
        return out


class HitCounter(Module):
    """Prints the number of hits"""

    def process(self, blob):
        try:
            self.cprint("Number of hits: {0}".format(len(blob["Hit"])))
        except KeyError:
            pass
        return blob


class HitCalibrator(Module):
    """A very basic hit calibrator, which requires a `Calibration` module."""

    def configure(self):
        self.input_key = self.get("input_key", default="Hits")
        self.output_key = self.get("output_key", default="CalibHits")

    def process(self, blob):
        if self.input_key not in blob:
            self.log.warn("No hits found in key '{}'.".format(self.input_key))
            return blob
        hits = blob[self.input_key]
        chits = self.calibration.apply(hits)
        blob[self.output_key] = chits
        return blob


class BlobIndexer(Module):
    """Puts an incremented index in each blob for the key 'blob_index'"""

    def configure(self):
        self.blob_index = 0

    def process(self, blob):
        blob["blob_index"] = self.blob_index
        self.blob_index += 1
        return blob


class StatusBar(Module):
    """Displays the current blob number."""

    def configure(self):
        self.iteration = 1

    def process(self, blob):
        prettyln("Blob {0:>7}".format(self.every * self.iteration))
        self.iteration += 1
        return blob

    def finish(self):
        prettyln(".", fill="=")


class TickTock(Module):
    """Display the elapsed time.

    Parameters
    ----------
    every: int, optional [default=1]
        Number of iterations between printout.
    """

    def configure(self):
        self.t0 = time()

    def process(self, blob):
        t1 = (time() - self.t0) / 60
        prettyln("Time/min: {0:.3f}".format(t1))
        return blob


class MemoryObserver(Module):
    """Shows the maximum memory usage

    Parameters
    ----------
    every: int, optional [default=1]
        Number of iterations between printout.
    """

    def process(self, blob):
        memory = peak_memory_usage()
        prettyln("Memory peak: {0:.3f} MB".format(memory))
        return blob


class Siphon(Module):
    """A siphon to accumulate a given volume of blobs.

    Parameters
    ----------
    volume: int
      number of blobs to hold
    flush: bool
      discard blobs after accumulation

    """

    def configure(self):
        self.volume = self.require("volume")  # [blobs]
        self.flush = self.get("flush", default=False)

        self.blob_count = 0

    def process(self, blob):
        self.blob_count += 1
        if self.blob_count > self.volume:
            log.debug("Siphone overflow reached!")
            if self.flush:
                log.debug("Flushing the siphon.")
                self.blob_count = 0
            return blob


class MultiFilePump(kp.Module):
    """Use the given pump to iterate through a list of files.

    Parameters
    ----------
    pump: Pump
      The pump to be used to generate the blobs.
    filenames: iterable(str)
      List of filenames.
    reindex: boolean
      Reindex the group_id by counting from 0 and increasing it continuously,
      this makes sure that the group_id is unique for each blob, otherwise
      the pump will usually reset it to 0 for every new file.

    """

    def configure(self):
        self.pump = self.require("pump")
        self.filenames = self.require("filenames")
        self.reindex = self.get("reindex", default=True)
        self.blobs = self.blob_generator()
        self.cprint("Iterating through {} files.".format(len(self.filenames)))
        self.n_processed = 0
        self.group_id = 0

    def blob_generator(self):
        for filename in self.filenames:
            self.cprint("Current file: {}".format(filename))
            pump = self.pump(filename=filename)
            for blob in pump:
                if self.reindex:
                    self._set_group_id(blob)
                blob["Filename"] = filename
                yield blob
                self.group_id += 1
            self.n_processed += 1

    def _set_group_id(self, blob):
        for key, entry in blob.items():
            if isinstance(entry, kp.Table):
                if hasattr(entry, "group_id"):
                    entry.group_id = self.group_id
                else:
                    blob[key] = entry.append_columns("group_id", self.group_id)

    def process(self, blob):
        return next(self.blobs)

    def finish(self):
        self.cprint(
            "Fully processed {} out of {} files.".format(
                self.n_processed, len(self.filenames)
            )
        )


class LocalDBService(kp.Module):
    """Provides a local sqlite3 based database service to store information"""

    def configure(self):
        self.filename = self.require("filename")
        self.thread_safety = self.get("thread_safety", default=True)
        self.connection = None

        self.expose(self.create_table, "create_table")
        self.expose(self.table_exists, "table_exists")
        self.expose(self.insert_row, "insert_row")
        self.expose(self.query, "query")

        self._create_connection()

    def _create_connection(self):
        """Create database connection"""
        try:
            self.connection = sqlite3.connect(
                self.filename, check_same_thread=self.thread_safety
            )
            self.cprint(sqlite3.version)
        except sqlite3.Error as exception:
            self.log.error(exception)

    def query(self, query):
        """Execute a SQL query and return the result of fetchall()"""
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def insert_row(self, table, column_names, values):
        """Insert a row into the table with a given list of values"""
        cursor = self.connection.cursor()
        query = "INSERT INTO {} ({}) VALUES ({})".format(
            table, ", ".join(column_names), ",".join("'" + str(v) + "'" for v in values)
        )
        cursor.execute(query)
        self.connection.commit()

    def create_table(self, name, columns, types, overwrite=False):
        """Create a table with given columns and types, overwrite if specified


        The `types` should be a list of SQL types, like ["INT", "TEXT", "INT"]
        """
        cursor = self.connection.cursor()
        if overwrite:
            cursor.execute("DROP TABLE IF EXISTS {}".format(name))

        cursor.execute(
            "CREATE TABLE {} ({})".format(
                name, ", ".join(["{} {}".format(*c) for c in zip(columns, types)])
            )
        )
        self.connection.commit()

    def table_exists(self, name):
        """Check if a table exists in the database"""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT count(name) FROM sqlite_master "
            "WHERE type='table' AND name='{}'".format(name)
        )
        return cursor.fetchone()[0] == 1

    def finish(self):
        if self.connection:
            self.connection.close()


def format_pathstring(instring):
    instring = instring.replace("(", "['").replace(")", "']")
    if instring.count(":")==1:
        instring = instring.replace(":","[")+"]"
    elif instring.count(":")>1:
        firstone = True
        while instring.find(":")>-1:
            if firstone:
                instring = instring.replace(":","[", 1)
                firstone = False
            else:
                instring = instring.replace(":","][")
        instring = instring + "]"
    return instring


class TableCollector(kp.Module):
    """Retrieves entries from blob and stores them in kp.Tables
    
    Parameters
    ----------
    parameters: str
        json formatted dictionary holding the selected parameters.
        format: {parameter_name, object_path, ...}
            * parameter_name can be freely chosen as identifier of the parameter, will become the name of the column in the outfile
            * object_path is the path to the entry in the blob encoded as e.g. '(event).tracks.E:0' for blob[event].tracks.E[0]
    """
    
    def configure(self):
        self.parameterpicker = json.loads(self.get("parameters", default=""))
        
    def process(self, blob):
        paramdict = {}
        typedict = {}
        for param in self.parameterpicker:
            pinfo = self.parameterpicker[param]
            if pinfo.find("(")>-1:
                if type(pinfo) is dict:
                    continue
                execstring = format_pathstring(pinfo)      
                value = eval("blob"+execstring)
                paramdict.setdefault(param, value)
            else:
                if pinfo in blob:
                    paramdict.setdefault(param, blob[pinfo])
                else:
                    paramdict.setdefault(param, np.nan)

        table = kp.Table(paramdict, h5singleton=False, h5loc="/events")
        blob["Eventlist"] = table
        return blob


class ValueProcessor(kp.Module):
    """Calculates new values from given entries in the blob and stores them in the blob
    
    Parameters
    ----------
    instructions: str
        json formatted dictionary holding the selected parameters.
        format: {processed_name: 
                    {'parameters': {param_name: object_path, ...},
                    {'description': some_more_info-optional},
                    {'expression': some_expression},
                    {'imports: {short: modulepath, ...}}
            * processed_name can be freely chosen as identifier of the new parameter, will become the name of entry in the blob
            * param_name can be freely chosen as identifier of the parameter within the expression - try to chose unique names, e.g. x_param
            * object_path is the path to the entry in the blob encoded as e.g. '(event).tracks.E:0' for blob[event].tracks.E[0]
            * use description to help understand the parameter - is optional
            * expression is a pythonian expression using the param_names instead of the parameter, e.g. 'x_param + y_param + numpy.pi'
            * imports holds the modulenames to import and their shortname in the expression, e.g. {'np':'numpy'} for 'import numpy as np'
    """
    
    def configure(self):
        self.processor = json.loads(self.get("instructions", default=""))
        self.imports = {}
        
        for process in self.processor:
            if "imports" in self.processor[process]:
                modules = {}
                for modname in self.processor[process]["imports"]:
                    impstring = self.processor[process]["imports"][modname]
                    module = importlib.import_module(impstring)
                    modules.setdefault(modname, module)
                self.imports.setdefault(process, modules)
                        
        
    def process(self, blob):
        for process in self.processor:
            
            params = {}
            parameters = self.processor[process]["parameters"]
            for param in parameters:
                execstring = format_pathstring(parameters[param])  
                val = eval("blob"+execstring)
                params.setdefault(param, val) 
                
            expression = self.processor[process]["expression"]
            for p in params:
                expression = expression.replace(p, str(params[p]))
            modules = {}
            if process in self.imports:
                modules = self.imports[process]
                
            blob[process] = np.nan
            try:
                if modules:
                    value = eval(expression, modules)
                else:
                    value = eval(expression)
                if type(value) is str:
                    value = value.encode()
            except:
                self.log.warn("Could not evaluate expression %s", expression)
            blob[process] = value

        return blob
    

class MetaAdder(kp.Module):
    """Add additional metadata to to the file.
       
       For now, the provenance from root files will be added, needs to be transfered to km3io
       Needs a 'rootfilepath' to read the info.
    """

    def configure(self):
        self.headerinfo = self.get("header", default="")
        self.hiddeninfo = self.get("hidden", default="")
        self.provpath = self.get("rootfilepath", default="")
        
        self.header = None
        self.hidden = {}
        if self.headerinfo:
            topmeta = json.loads(self.headerinfo)
            for key in topmeta:
                if type(topmeta[key]) is str:
                    topmeta[key] = topmeta[key].encode()
            self.header = kp.Table(topmeta, h5singleton=True, h5loc="/header")
        if self.hiddeninfo:
            self.hidden = json.loads(self.hiddeninfo)
            
        self.expose(self.hidden, "hidden_metadata")
        
        jppprov = _jpp_info_from_file(self.provpath)
        workflow = Workflow.from_stages(jppinfo = jppprov)
        self.hiddeninfo.setdefault("jpp_prov", workflow.get_dicts())
        
    def process(self, blob):
        blob["Header"] = self.header
        return blob        


class EventSelector(kp.Module):
    """Select events according to criterion and keep track of discarded events
    
    Parameters
    ----------
    selection: str
        json formatted dictionary holding the selection criteria.
        format: {criterion_name :{'parameter': object_path, 'constraint': expression}}
            * criterion_name can be freely chosen as identifier of the criterion
            * object_path is the path to the entry in the blob encoded as e.g. '(event).tracks.E:0' for blob[event].tracks.E[0]
            * expression is a pythonian expression including an 'x' instead of the parameter, e.g. 'x > 200'.

    recordparameter: list
        list of object paths of parameters to record if event is dropped
    """
    
    def configure(self):
        self.selection = json.loads(self.get("selection", default=[]))
        self.recordparams = json.loads(self.get("recordparameters", default=[]))
        self.discarded = {}
        self.counter = 0
        
        self.expose(self.discarded, "discarded_blob_record")
        
    def process(self, blob):
        failes = []
        for skey in self.selection:
            selector = self.selection[skey]
            execstring = format_pathstring(selector["parameter"])  
            param = eval("blob"+execstring)
            evalstring = selector["constraint"].replace("x", str(param))
            testresult = eval(evalstring)
            if not testresult:
                failes.append(skey)

        if failes:
            self.counter += 1
            record = {}

            for param in self.recordparams:
                execstring = format_pathstring(param)  
                value = eval("blob"+execstring)
                record.setdefault(param, value)
            self.discarded.setdefault(str(self.counter),
                                      {"fails": failes, 
                                       "record": record
                                       })
        else:
            return blob

                
            
            
