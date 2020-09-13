# Filename: sortprov.py
"""
Classes and functions for processing provenance information
"""
import uproot
import uuid
import json
from copy import deepcopy
from datetime import datetime

# Definition of provenance classes
class Activity: # merging ActivityDescription into class
    def __init__(self, kid = "unset", name = "", startTime = "", endTime = "", execution = {}):
        self.kid = kid
        self.name = name
        self.startTime = startTime
        self.endTime = endTime
        self.execution = execution # this describes the software
        
    @classmethod
    def from_dict(cls, indict):
        ws = cls()
        for key in indict:
            setattr(ws, key, indict[key])
        return ws
        
class Configuration: # merging ConfigFileDescription
    # mushy version of config files & parameter setting
    def __init__(self, name = "", parametersetting = {}, system = {}):
        self.name = name
        self.parametersetting = parametersetting # config file settings
        self.system = system # extending configuration to environment
        
    @classmethod
    def from_dict(cls, indict):
        ws = cls()
        for key in indict:
            setattr(ws, key, indict[key])
        return ws
        
class Entity: 
    def __init__(self, kid = "unset", name="", location="", generatedAtTime="", description = ""):
        self.kid = kid
        self.name = name
        self.location = location
        self.generatedAtTime = generatedAtTime
        self.description = description
        
    @classmethod
    def from_dict(cls, indict):
        ws = cls()
        for key in indict:
            setattr(ws, key, indict[key])
        return ws

class WorkflowStep: # should record 
    def __init__(self, kid = "unset", inputs = [], 
                 outputs = [], 
                 configuration = Configuration(), 
                 activity = Activity()):
        self.kid = kid
        self.inputs = inputs # List of entities
        self.outputs = outputs # List of entities
        self.configuration = configuration
        self.activity = activity
        
    @classmethod
    def from_dict(cls, indict):
        ws = cls()
        for key in indict:
            if key == "configuration":
                setattr(ws, key, Configuration.from_dict(indict[key]))
            elif key == "activity":
                setattr(ws, key, Activity.from_dict(indict[key]))
            else:
                setattr(ws, key, indict[key])
        return ws
        
    def get_ref(self, include_params = False, include_software = False):
        # create WorkflowStepRef from from workflow step
    
        ref = WorkflowStepRef(refID = self.kid, name = self.activity.name)
        if include_params:
            ref.settings = self.configuration.parametersetting
        if include_software:
            ref.settings = self.activity.execution
        return ref
    
    def get_dict(self):
        outdict = {}
        for key in ("kid", "inputs", "outputs", "configuration", "activity"):
            info = make_dict_from_object(getattr(self, key))
            outdict.setdefault(key, info)
        return outdict
        
class WorkflowStepRef:
    def __init__(self, refID = "", name= "", settings = None, software = None):
        self.refID = refID
        self.name = name # optional, activity name
        self.settings = settings
        self.software = software
        
        
class Workflow:
    def __init__(self, kid = "", name = ""):
        self.kid = kid
        self.name = name
        self.workflowsteps = []
        
    @classmethod
    def from_stages(cls, jppinfo = [], pipeinfo = []):
        wf = cls()
        
        wf.kid = uuid.uuid1()
        if jppinfo:
            wf.add_steps(jppinfo, match_jpp_provstep)
        if pipeinfo:
            wf.add_steps(pipeinfo, match_km3pipe_provstep)
        return wf
    
    @classmethod
    def from_dict(cls, indict):
        ws = cls()
        for key in indict:
            if key == "workflowsteps":
                for entry in indict[key]:
                    entry = WorkflowStep.from_dict(entry)
            else:
                setattr(ws, key, indict[key])
        return ws
                
    def get_workflowreflist(self, include_params = False, include_software = False):
        reflist = []
        for step in self.workflowsteps:
            ref = step.get_ref(include_params, include_software)
            reflist.append(make_dict_from_object(ref))
        return reflist
    
    def get_dicts(self):
        outlist = []
        print (len(self.workflowsteps))
        for step in self.workflowsteps:
            outlist.append(make_dict_from_object(step))
        return outlist
    
    def add_steps(self, stepinfos = [], decoder = None):
        for step in stepinfos:
            newstep = step
            if decoder: 
                newstep = decoder(newstep)
            self.workflowsteps.append(deepcopy(newstep))
        

def match_jpp_provstep(indict):
    # reading jpp-formatted provenance list, returning workflow step
    
    step = WorkflowStep()
    step.kid = uuid.uuid1()
    step.activity.name = indict["application"]
    step.activity.execution["version"] = indict["GIT"]
    step.activity.execution["package"] = "jpp"
    step.configuration.name = "JPP_"+indict["application"]
    step.configuration.system["namespace"] = indict["namespace"] #whatever that means
    pos = indict["system"].find(":")
    
    dt = datetime.strptime(indict["system"][pos-13:pos+15].lstrip(" "), 
                           "%a %b %w %H:%M:%S UTC %Y")
    step.activity.startTime = dt.isoformat()
    step.configuration.system["setting"] = indict["system"][pos-13]
    command = indict["command"].split(" -")
    step.activity.execution["basecommand"] = command[0]
    coms = {}
    for line in command[1:len(command)]:
        argument = line[2:len(line)]
        filename = argument
        if argument.count("/")>-1:
            filename = argument[max(argument.rfind("/"), 0):len(argument)]
        if line[0]=="f":
            step.inputs.append(deepcopy(Entity(name=filename, location=argument)))
        elif line[0]=="a":
            step.inputs.append(deepcopy(Entity(name=filename, location=argument)))
        elif line[0]=="o":
            step.outputs.append(deepcopy(Entity(name=filename, location=argument)))
        if line[0:1] != "-!":
            coms.setdefault("-"+line[0], argument)
    step.configuration.parametersetting = coms     
            
    return step


def match_km3pipe_provstep(indict):
    # reading km3pipe provenance list, returning workflow step
    
    step = WorkflowStep()
    step.kid = indict["uuid"]
    step.activity.name = indict["name"]
    step.activity.startTime = indict["start"]["time_utc"]
    step.activity.stopTime = indict["stop"]
    if indict["input"]:
        for infile in indict["input"]:
            fullname = infile["url"]
            filename = fullname
            if fullname.count("/")>-1:
                filename = fullname[max(fullname.rfind("/"), 0):len(fullname)]
            step.inputs.append(deepcopy(Entity(name=filename, location=fullname, description = infile["comment"])))
    if indict["output"]:
        for infile in indict["output"]:
            fullname = infile["url"]
            filename = fullname
            if fullname.count("/")>-1:
                filename = fullname[max(fullname.rfind("/"), 0):len(fullname)]
            step.inputs.append(deepcopy(Entity(name=filename, location=fullname, description = infile["comment"])))
    step.configuration.system = indict["system"]
    step.activity.execution.setdefault("status", indict["status"])
    step.activity.execution.setdefault("duration", indict["duration"])
    step.configuration.parametersetting = indict["configuration"]

    return step


def make_dict_from_object(inobject):
    # converting worflow step to dictionary

    outdict = {}

    a = inobject.__dict__
    for key in a:
        val = ""
        if type(a[key]) is list:
            newlist = []
            for thisone in a[key]:
                newlist.append(thisone.__dict__)
            val = newlist
        elif type(a[key]) in (Activity, Configuration, WorkflowStep):
            val = a[key].__dict__
        else:
            val = a[key]
        outdict.setdefault(key, val)
    return outdict


def _jpp_info_from_file(rootfilepath):
    # read provenance info directly from a root file, returns dictionary

    metas = []

    with uproot.open(rootfilepath) as f:
        for key in f["META"]:
            if key.decode().find("JMeta")>-1:
                metas.append([key, f["META"][key]._fTitle])
        metas.sort()
        
        fullprov = []
        for meta in metas:
            entry = f["META"][meta[1]]
            info = entry._fTitle.decode()
            step = {}
            if "\n" in info:
                for l in info.split("\n"):
                    if "=" in l:
                        step.setdefault(l.split("=")[0], l.split("=")[1])
            else:
                step = info
            fullprov.append(step)
    return fullprov

def _read_info_from_prov(provinfo):
    # get provenance info from km3pipe either from file or from class

    if type(provinfo) is str:
        with open(provinfo, "r") as f:
            provinfo = json.loads(f.read())
            
    return provinfo
