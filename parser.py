import pandas as pd
from parameters import *
from clade import *

class Parser:
    ''' class for parsing input parameters provided in CSV input file '''

    ###############################################
    @classmethod
    def parseCSVInput(cls, csv_fname: str) -> None:
        ''' class-level method to parse input parameters from CSV input file
        Parameters:
            csv_fname: filename of the CSV input file (str)
        '''
        # first, fish out the class-level variables from the Parameters class,
        # creating a list of those variable names; we will create a simple
        # string below to exec thereby assigning the Parameters variables as
        # we find the corresponding entry in the input CSV
        sim_params_list = [attr for attr in dir(Parameters) \
                           if not attr.startswith("__") and \
                              not callable(getattr(Parameters, attr))]

        # lambda to morph a Clade instance variable into its corresponding
        # Clade setter name; this will convert something like
        # _avg_residence_time into Clade.setAvgResidenceTime (with a few
        # replace exceptions that title() doesn't handle well)
        morph = lambda var: 'Clade.set' + \
            var.title().replace('Ppr','PPR').replace('Mcr','MCR').replace('G1Sg2M','G1SG2M').replace('_','')

        # now fish out the instance variables from Clade, creating a dictionary
        # with key as the variable entry in the CSV input file, and value as the
        # corresponding setter inside Clade (see eval below and more lambda);
        # with this dictionary, when we encounter clade-level parameters in the
        # CSV input file, we can find the entry (key will match variable name in
        # the CSV input file) and then call the method (key's value in dict)
        # passing the parameter value given in the CSV input file
        # (ignore the clade_object class-level variable, anything starting with
        #  '__' (e.g., __str__), and any callable function/method)
        clade_methods_dict = { \
            attr.upper()[1:]:eval(morph(attr)) for attr in dir(Clade) \
               if attr != "clade_objects" and not attr.startswith("__") and \
                  not callable(getattr(Clade, attr))}

        # read the input CSV as a pandas dataframe 
        csv_dataframe = pd.read_csv(csv_fname, header = 0)

        # drop (ignore) any blank rows in the CSV (NaN in all entries)
        csv_dataframe = csv_dataframe.dropna(subset="Parameter Name")

        clade_number = 0 # used below to track which clade to update
        clade = None

        for r in range(len(csv_dataframe)):
            row = csv_dataframe.iloc[r]
            # each row should be of the form: parameter name, value, description
            # bgl: Feb|Mar 2024
            parameter_name = row.iloc[0]
            value          = row.iloc[1]
            description    = row.iloc[2]
            #parameter_name = row[0]
            #value          = row[1]
            #description    = row[2]

            # ignore any row where the parameter name begins with # (comment)
            if isinstance(parameter_name, str) and parameter_name.startswith('#'): continue

            # convert value to either int or float; if unsuccessful, leave as
            # str if filename, otherwise eval (e.g., for tuples)
            try: value = int(value)
            except:
                try: value = float(value)
                except:
                    if parameter_name in ('POPULATION_FILENAME','CSV_FILENAME','LOG_FILENAME','INITIAL_PLACEMENT','GRID_TYPE'):
                        value = repr(value)  # repr includes quotes for str
                    else:
                        # str like "(0.5,0.5)" will be eval'd to tuple
                        if value.lower() == "true":  value = "True"
                        if value.lower() == "false": value = "False"
                        value = eval(str(value))

            if parameter_name in sim_params_list:
                # simulation-level parameter -- use exec to just set its value
                exec_str = f'Parameters.{parameter_name} = {value}'
                exec(exec_str)
            else:
                # clade-level parameter
                if parameter_name == "CLADE_NUMBER":
                    # add the previous clade to the list of clades in Clade
                    if clade is not None: Clade.addClade(clade)
                    # then create a new clade
                    clade_number += 1
                    clade = Clade(clade_number)
                assert(clade is not None)
                # get the setter method reference and then call, passing value
                setter_method = clade_methods_dict[parameter_name]
                setter_method(clade, value)  # pass in clade for self

        # add the last clade (still in progress)
        assert(clade is not None)
        Clade.addClade(clade)

        ## check whether user want to see all parameter values 
        if Parameters.PRINT_PARAMETER_VALUES:
            Parameters.printParameters()
            for clade in Clade.clade_objects:
                print(clade)
