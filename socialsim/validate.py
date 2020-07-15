import json
import numpy as np
import pandas as pd
from .load import load_data

VALID_OPTIONS = {'cp4': {'simulation_periods': {"february1-february14", "february8-february21",
                                                "february15-february28", "february22-february28",
                                                "march1-march14", "march8-march21", "march15-march28",
                                                "march22-april4"},
                         'simulation_windows': {"february1-february14": ['02-01', '02-14'],
                                                "february8-february21": ['02-08', '02-21'],
                                                "february15-february28": ['02-15', '02-28'],
                                                "february22-february28": ['02-22', '02-28'],
                                                "march1-march14": ['03-01', '03-14'],
                                                "march8-march21": ['03-08', '03-21'],
                                                "march15-march28": ['03-15', '03-28'],
                                                "march22-april4": ['03-22', '04-04']},
                         'teams': {'pnnl', 'ucf-garibay', 'uiuc', 'usc', 'usf', 'uva'},
                         'platforms': {'twitter', 'youtube'},
                         'actiontypes': {'tweet', 'retweet', 'quote', 'reply', 'comment', 'video'}
                         }
                 }

def check_meta(submission_filepath, challenge):
    try:
        # test that meta is valid json object
        with open(submission_filepath, 'r') as f:
            submission_meta = f.readline()
        submission_meta = json.loads(submission_meta)
    except Exception as e:
        return f'Metadata is not a valid json object.\n\tjson.loads Exception: {str(e)}'

    return check_metadata_object(submission_meta, challenge)


def check_metadata_object(submission_meta, challenge):
    teams = ['pnnl', 'ucf-garibay', 'uiuc', 'usc', 'usf', 'uva']
    simulation_periods = VALID_OPTIONS[challenge]['simulation_periods']
    meta_issues, meta_warnings = [], []
    # test required fields are valid
    model_identifier = str(submission_meta.get('model_identifier', 'UNKNOWN'))
    simulation_period = str(submission_meta.get('simulation_period', 'UNKNOWN'))
    team = str(submission_meta.get('team', 'UNKNOWN'))
    # check team is valid
    if team not in teams:
        valid_teams = ', '.join([f'"{x}"' for x in teams])
        err_msg = ''
        if team == 'UNKNOWN':
            err_msg = f'"team" field is missing.'
        else:
            err_msg = f'"team" field is invalid: {team} is not a valid team.'
        meta_issues.append(f'{err_msg}\n\tValid team options are: {valid_teams}')
    # check simulation_period is valid
    if simulation_period not in simulation_periods:
        valid_simulation_periods = ', '.join([f'"{x}"' for x in simulation_periods])
        err_msg = ''
        if simulation_period == 'UNKNOWN':
            err_msg = f'"simulation_period" field is missing.'
        else:
            err_msg = f'"simulation_period" field is invalid: {simulation_period} is not a valid scenario.'
        meta_issues.append(f'{err_msg}\n\tValid scenario options are: {valid_simulation_periods}')
    # check model_identifier is valid
    if model_identifier in ['UNKNOWN', None, np.nan] or type(model_identifier) is not str:
        err_msg = ''
        if model_identifier == 'UNKNOWN':
            err_msg = f'"model_identifier" field is missing.'
        else:
            err_msg = f'"model_identifier" field is invalid: {model_identifier} is not a valid model identifier.'
        meta_issues.append(f'{err_msg}\n\tValid model_identifier options are string values.')

    # test that the submission_meta object is not a record
    if 'nodeID' in submission_meta.keys():
        fields = ','.join(submission_meta.keys())
        meta_warnings.append(
            f'Metadata may be missing from submission file, metadata object looks like an event record.\n\tFields are: {fields}')

    if len(meta_issues) > 0:
        meta_status = 'ERRORS:\n\t' + '\n\n\t'.join(meta_issues) + '\n\nWARNINGS:\n\t' + '\n\t'.join(meta_warnings)
    else:
        # if header is a valid json object and includes all the required fields, header_status is successful
        meta_status = 'success'

    validation_flag = len(meta_issues) == 0
    return meta_status, simulation_period, validation_flag


def check_all_present(valid_items, subm_items, item_type):
    errors, warnings = [], []
    ## check that submission contains at least some of the groundtruth items, if not raise error
    if subm_items.intersection(valid_items) == set():
        errors.append(
            f'Submission does not include any of the valid {item_type} options ({valid_items}).\n\tSubmission only includes data for: {subm_items}')
    ## check whether submission contains all of the groundtruth platforms, if not raise warning
    if subm_items != valid_items:
        missing_items = valid_items - subm_items
        warnings.append(
            f'Submission does not contain all of the valid {item_type} options.\n\tSubmission does not include data for: {missing_items}')
        ## check whether submission contains only the groundtruth platforms, if not raise warning
    extra_items = subm_items - valid_items
    if extra_items != set():
        warnings.append(f'Submission includes {item_type} that are not in the valid {item_type} options: {extra_items}')
    return errors, warnings


def check_records(submission_filepath, nodelist, simulation_period, challenge):
    errors, warnings = [], []
    try:
        # test that submission file can be loaded
        subm = load_data(submission_filepath, ignore_first_line=True, verbose=False)
        loaded = True
    except Exception as e:
        errors.append('Submission could not be loaded: ' + str(e))
        loaded = False

    if loaded:
        # platform tests
        valid_items = VALID_OPTIONS[challenge]['platforms']
        subm_items = set(subm['platform'].unique())
        platform_errors, platform_warnings = check_all_present(valid_items, subm_items, 'platforms')
        errors.extend(platform_errors)
        warnings.extend(platform_warnings)

        if nodelist is not None:
            if len(nodelist) > 0:
                # informationID tests
                subm_items = set(subm['informationID'].unique())
                informationID_errors, informationID_warnings = check_all_present(nodelist, subm_items, 'informationIDs')
                errors.extend(informationID_errors)
                warnings.extend(informationID_warnings)
            else:
                warnings.append('Node list passed to valdiate against is empty so validation did not check for missing/extra informationIDs.')

        # test that there are no NaN items in required event details
        for c in ['informationID', 'nodeTime', 'nodeID', 'parentID', 'rootID', 'platform', 'actionType', 'nodeUserID']:
            if len(subm[c]) != len(subm[c].dropna()):
                errors.append(f'{c} can not be NaN values.')
                print(c)
                print(subm[c].astype(str).unique())
        # check for empty user-user network
        parentID_nodeID_overlap = set(subm['parentID']).intersection(set(subm['nodeID']))
        if len(parentID_nodeID_overlap) == 0:
            warnings.append(
                'There is no overlap between nodeID values and parentID values -- the user-to-user network created from this submission will be empty.')

        # check that nodeTimes fall within simulation window
        try:
            simulation_window = VALID_OPTIONS[challenge]['simulation_windows'][simulation_period]
            minday = f'2019-{simulation_window[0]}'
            maxday = f'2019-{simulation_window[1]} 23:59:59'
            maxday_str = maxday.split(' ')[0]
            subm['nodeTime'] = pd.to_datetime(subm['nodeTime']).astype(str)
            subm_minday, subm_maxday = subm['nodeTime'].min(), subm['nodeTime'].max()
            if subm_maxday <= minday:
                errors.append(
                    f'There is no data within the simulation period, all nodeTime values occur before the simulation period ({minday} - {maxday_str}).\n\tSubmission nodeTime values -- Min: {subm_minday} Max: {subm_maxday}')
            elif subm_minday > maxday:
                errors.append(
                    f'There is no data within the simulation period, all nodeTime values occur after the simulation period ({minday} - {maxday_str}).\n\tSubmission nodeTime values -- Min: {subm_minday} Max: {subm_maxday}')
            elif subm_minday < minday:
                warnings.append(
                    f'Some events occur before the simulation period, earliest nodeTime value is {subm_minday}')
            elif subm_maxday > maxday:
                warnings.append(
                    f'Some events occur after the simulation period, latest nodeTime value is {subm_maxday}')
        except Exception as e:
            warnings.append('Could not validate nodeTimes occur within the simulation period: ' + str(e))

        # actionType tests
        valid_items = VALID_OPTIONS[challenge]['actiontypes']
        subm_items = set(subm['actionType'].unique())
        platform_errors, platform_warnings = check_all_present(valid_items, subm_items, 'actionType')
        errors.extend(platform_errors)
        warnings.extend(platform_warnings)

    result = ''

    if len(errors) > 0:
        result = result + 'ERRORS:\n\t'
        result = result + '\n\n\t'.join(errors) + '\n\n'
    if len(warnings) > 0:
        result = result + 'WARNINGS:\n\t'
        result = result + '\n\n\t'.join(warnings)

    if result == '': result = 'success'

    validation_flag = len(errors) == 0
    return result, validation_flag


def validation_flag(submission_filepath, challenge='cp4', nodelist_filepath=None):
    ## check metadata
    meta_status, simulation_period, meta_validation_flag = check_meta(submission_filepath, challenge)

    ## check event records
    try:
        if nodelist_filepath is None:
            nodelist = None
        else:
            with open(nodelist_filepath, 'r') as f:
                nodelist = set([x.strip() for x in f.readlines()])
    except Exception as e:
        pass

    records_status, records_validation_flag = check_records(submission_filepath, nodelist, simulation_period)

    if meta_validation_flag and records_validation_flag:
        validation_flag = True
    else:
        validation_flag = False

    return validation_flag


def validation_report(submission_filepath, challenge='cp4', nodelist_filepath=None):

    print('\n\nValidating ' + str(submission_filepath) + ' ...')
    validation_report = ['Submission: ' + str(submission_filepath)]
    ## check metadata
    meta_status, simulation_period, meta_validation_flag = check_meta(submission_filepath, challenge)
    validation_report.append('\n------------Metadata------------')
    if meta_status != 'success':
        validation_report.append(meta_status)
    else:
        validation_report.append('Validated without issue')
    validation_report.append('--------------------------------')

    ## check event records
    validation_report.append('\n-------------Events-------------')
    try:
        if nodelist_filepath is None:
            nodelist = None
            validation_report.append(
                '*** No nodelist filepath was specified so validation did not check for missing/extra informationIDs. ***\n')
        else:
            with open(nodelist_filepath, 'r') as f:
                nodelist = set([x.strip() for x in f.readlines()])
    except Exception as e:
        validation_report.append('Error loading nodelist from file. ' + str(e))

    records_status, records_validation_flag = check_records(submission_filepath, nodelist, simulation_period, challenge)
    if records_status != 'success':
        validation_report.append(records_status)
    else:
        validation_report.append('Validated without issue')
    validation_report.append('--------------------------------\n')

    validation_report = '\n'.join(validation_report)
    if meta_validation_flag and records_validation_flag:
        validation_flag = True
    else:
        validation_flag = False

    return validation_flag, validation_report