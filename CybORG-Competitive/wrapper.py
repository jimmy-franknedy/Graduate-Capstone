# Import Cardiff Wrappers here
import inspect
from cardiff import *
from cardiff.cage2.Wrappers.BlueTableWrapper import BlueTableWrapper
from cardiff.cage2.Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from CybORG.Agents import B_lineAgent, SleepAgent

from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import TrueTableWrapper
from CybORG.Shared.Results import Results
from CybORG.Shared.Actions import (
    Monitor,
    Analyse,
    Remove,
    Restore,
    DecoyApache,
    DecoyFemitter,
    DecoyHarakaSMPT,
    DecoySmss,
    DecoySSHD,
    DecoySvchost,
    DecoyTomcat,
    DecoyVsftpd,
    PrivilegeEscalate,
    ExploitRemoteService,
    DiscoverRemoteSystems,
    Impact,
    DiscoverNetworkServices,
    Sleep,
)

from itertools import product
from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

class CompetitiveWrapper(BaseWrapper):
    def __init__(self, turns, env=None, agent=None, output_mode="vector", cardiff=False):
        super().__init__(env, agent)
        self.env = TrueTableWrapper(env=env, agent=agent)

        # Used for training against cardiff agent
        self.cardiff_flag=cardiff

        self.agent = agent

        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.output_mode = output_mode
        self.success = None

        self.subnets = "Enterprise", "Op", "User"
        self.hostnames = (
            "Enterprise0",  # 0
            "Enterprise1",  # 1
            "Enterprise2",  # 2
            "Op_Host0",     # 4
            "Op_Host1",     # 5
            "Op_Host2",     # 6
            "Op_Server0",   # 3
            "User1",        # 7
            "User2",        # 8
            "User3",        # 9
            "User4",        # 10
        )

        blue_lone_actions = [["Monitor"]]  # actions with no parameters
        blue_host_actions = (
            "Analyse",
            "Remove",
            "Restore",
            "DecoyApache", 
            "DecoyFemitter", 
            "DecoyHarakaSMPT", 
            "DecoySmss", 
            "DecoySSHD", 
            "DecoySvchost", 
            "DecoyTomcat", 
            "DecoyVsftpd",
        )  # actions with a hostname parameter
        red_lone_actions = [["Sleep"], ["Impact"]]  # actions with no parameters
        red_network_actions = [
            "DiscoverSystems"
        ]  # actions with a subnet as the parameter
        red_host_actions = (
            "DiscoverServices",
            "ExploitServices",
            "PrivilegeEscalate",
        )
        self.blue_action_list = blue_lone_actions + list(
            product(blue_host_actions, self.hostnames)
        )
        # print("wrapper.py: ", len(self.blue_action_list)) # Blu should have 145 possible actions; currently has 122
        self.red_action_list = (
            red_lone_actions
            + list(product(red_network_actions, self.subnets))
            + list(product(red_host_actions, self.hostnames))
        )
        # print("wrapper.py: ", len(self.red_action_list)) # Red has 38 possible actions

        self.subnet_map = {}  # subnets are ordered [User, Enterprise, Op]
        self.ip_map = (
            {}
        )  # ip addresses are ordered ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender', 'Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'User0', 'User1', 'User2', 'User3', 'User4']

        self.host_scan_status = [0]*len(self.hostnames)
        self.subnet_scan_status = [0]*len(self.subnets)*2
        
        # see _create_red_vector for explanation of flags
        self.impact_status = [0]

        self.turns_per_game = turns
        self.turn = 0

        # Added for cardiff implementation
        self.cardiff_action_list = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                                    132, 2, 15, 24, 25, 26, 27]
        
        # defender specific actions are set to Monitor
        # defender specific actions are 132,2,15
        # if trained with complete 145 blue action; we can just remove this dictionary
        # if remove this dictionary; also remove logic in env_controller, resolve blue action
        self.cardiff_action_table = {133:23, 
                                    134:24, 
                                    135:25, 
                                    139:29, 
                                    3:1, 
                                    4:2, 
                                    5:3, 
                                    9:7, 
                                    16:12, 
                                    17:13, 
                                    18:14, 
                                    22:18, 
                                    11:8, 
                                    12:9, 
                                    13:10, 
                                    14:11, 
                                    141:30, 
                                    142:31, 
                                    143:32, 
                                    144:33,
                                    132:0, 
                                    2:0, 
                                    15:0, 
                                    24:19, 
                                    25:20, 
                                    26:21, 
                                    27:22,
                                    51:53,
                                    116:108,
                                    55:56,
                                    107:100,
                                    120:111,
                                    29:34,
                                    43:46,
                                    44:47,
                                    37:41,
                                    115:107,
                                    76:74,
                                    102:96,
                                    51:53,
                                    116:108,
                                    38:42,
                                    90:86,
                                    130:120,
                                    91:87,
                                    131:121,
                                    54:0,
                                    106:0,
                                    28:0,
                                    119:0,
                                    61:62,
                                    35:40,
                                    113:106,
                                    126:117,}

    def convert_blue_action(self, action):
        return self.cardiff_action_table[action]

    # convert the discrete action choice into its corresponding CybORG action
    def resolve_blue_action(self, action, cardiff=False):
        
        # assume a "single session" in the CybORG action space
        cyborg_space = self.get_action_space(agent="Blue")

        # print("Checking the cyborg_space")
        # print(cyborg_space)

        # Check how many sessions Blue has?
        # print("Blu sessions")
        # print(list(cyborg_space["session"]))

        # This is the part where the author only selects one session! I think we have a total of 13 sessions, but 11 usable sessions for both agents
        session = list(cyborg_space["session"].keys())[0]

        # *** ACTION CONFLICT ARISES FROM HERE *** 

        # Convert the cardiff action to correct CybORG action
        if(cardiff and (action > 0)):
            if action in self.cardiff_action_table:
                action = self.cardiff_action_table[action]
                # print("converted action is ", action)
            else:
                raise ValueError(f"Action: {action} has not been converted!")

        cyborg_action = self.blue_action_list[action]

        if cyborg_action[0] == "Analyse":
            # print("calling - Analyse on ", cyborg_action[1])
            return Analyse(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "Remove":
            # print("calling - Remove on ", cyborg_action[1])
            return Remove(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "Restore":
            # print("calling - Restore on ", cyborg_action[1])
            return Restore(hostname=cyborg_action[1], agent="Blue", session=session)

        elif cyborg_action[0] == "DecoyApache":
            # print("calling - DecoyApache on ", cyborg_action[1])
            return DecoyApache(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoyFemitter":
            # print("calling - DecoyFemitter on ", cyborg_action[1])
            return DecoyFemitter(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoyHarakaSMPT":
            # print("calling - DecoyHarakaSMPT on ", cyborg_action[1])
            return DecoyHarakaSMPT(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoySmss":
            # print("calling - DecoySmss on ", cyborg_action[1])
            return DecoySmss(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoySSHD":
            # print("calling - DecoySSHD on ", cyborg_action[1])
            return DecoySSHD(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoySvchost":
            # print("calling - DecoySvchost on ", cyborg_action[1])
            return DecoySvchost(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoyTomcat":
            # print("calling - DecoyTomcat on ", cyborg_action[1])
            return DecoyTomcat(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "DecoyVsftpd":
            # print("calling - DecoyVsftpd on ", cyborg_action[1])
            return DecoyVsftpd(hostname=cyborg_action[1], agent="Blue", session=session)

        else:
            # print("calling - Monitor on Defender?", cyborg_action[1])
            return Monitor(agent="Blue", session=session)
    
    def resolve_red_action(self, action):

        # assume a single session in the cyborg action space
        cyborg_space = self.get_action_space(agent="Red")

        # Print the action space
        # print("Red Agent Action Space",cyborg_space)

        # Print how many sessions red may have?
        # print("Red sessions")
        # print(list(cyborg_space["session"]))

        session = list(cyborg_space["session"].keys())[0]

        cyborg_action = self.red_action_list[action]
        if cyborg_action[0] == "Impact":
            # print("calling - Impact on Op_Server0", )
            return Impact(agent="Red", hostname="Op_Server0", session=session)
        elif cyborg_action[0] == "DiscoverSystems":
            # print("calling - Impact on ",self.subnet_map[cyborg_action[1]])
            return DiscoverRemoteSystems(
                subnet=self.subnet_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "DiscoverServices":
            # print("calling - DiscoverNetworkServices on ",self.subnet_map[cyborg_action[1]])
            return DiscoverNetworkServices(
                ip_address=self.ip_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "ExploitServices":
           # print("calling - ExploitServices on ",self.subnet_map[cyborg_action[1]])
            return ExploitRemoteService(
                ip_address=self.ip_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "PrivilegeEscalate":
            # print("calling - PrivilegeEscalate on ",self.subnet_map[cyborg_action[1]])
            return PrivilegeEscalate(
                hostname=cyborg_action[1], agent="Red", session=session
            )
        else:
            # print("calling sleep!")
            return Sleep()

    def map_network(self, env):

        # Confirmation
        # Check the current mapping of IP Addresses to hostnames
        # print("Current ip_map() is: ")
        # print(env.get_ip_map())
        # print("Done!")

        # Ensure that the ordering of the subnet and IP address are correct
        i = 0  # count through the networks to assign the correct IP
        for subnet in env.get_action_space(agent="Red")["subnet"]:
            self.subnet_map[self.subnets[i]] = subnet
            i += 1

        i = -1  # counter through the IP addresses to assign the correct hostname
        for address in env.get_action_space(agent="Red")["ip_address"]:
            # skip mapping the Defender client and User0
            # Defender is at index -1
            # Attacker is at index 7

            if 0 <= i and i < 7:
                self.ip_map[self.hostnames[i]] = address
                # Confirmation - Check to see mapping between IP Addresses to hostnames
                # print("i: ", i , "mapping ", self.hostnames[i], "to ", address)
            if  7 < i:
                self.ip_map[self.hostnames[i-1]] = address
                # Confirmation - Check to see mapping between IP Addresses to hostnames
                # print("i: ", i , "mapping ", self.hostnames[i-1], "to ", address)
            i += 1

    # returns the blue and red observation vectors
    def reset(self, cardiff=False):

        self.blue_info = {}
        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.success = None

        self.turn = 0

        result = self.env.reset()
        self.map_network(self.env)

        self.host_scan_status = [0]*len(self.hostnames)
        self.subnet_scan_status = [0]*len(self.subnets)*2
        self.impact_status = [0]

        blue_obs = result.blue_observation # the environment now returns both observations, so blue_observation needs to be specified here
        self._initial_blue_obs(blue_obs)
        blue_vector = self.blue_observation_change(blue_obs, baseline=True, cardiff=cardiff)

        red_obs = result.red_observation
        red_vector = self.red_observation_change(red_obs, self.get_last_action(agent="Red"))

        return (blue_vector, red_vector)
    
    def step(self, red_action, blue_action, cardiff=False) -> Results:
        # print("wrapper.py - step()")
        # print("cardiff is ",cardiff)


        red_step = self.resolve_red_action(red_action)
        # print("red_step is: ",red_step)

        blue_step = self.resolve_blue_action(blue_action, cardiff=cardiff)
        # print("blu_step is: ",blue_step)

        result = self.env.step(red_step, blue_step)
        self.turn += 1

        blue_obs = result.blue_observation
        blue_vector = self.blue_observation_change(blue_obs,cardiff=cardiff)
        result.blue_observation = blue_vector

        red_obs = result.red_observation
        red_vector = self.red_observation_change(red_obs, self.get_last_action(agent="Red"))
        result.red_observation = red_vector   

        result.action_space = self.action_space_change(result.action_space)
        # note this is the blue reward leaving the wrapper, red trainer must flip signal

        return result

    def _initial_blue_obs(self, obs):
        obs = obs.copy()
        self.baseline = obs
        del self.baseline["success"]
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            interface = host["Interface"][0]
            subnet = interface["Subnet"]
            ip = str(interface["IP Address"])
            hostname = host["System info"]["Hostname"]
            self.blue_info[hostname] = [str(subnet), str(ip), hostname, "None", "No"]
        return self.blue_info
    
    def blue_observation_change(self, observation, baseline=False, cardiff=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs["success"]

        self._process_blue_action(success)
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs["success"]
        # TODO check what info is for baseline
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = "None"
                info[host][-1] = "No"
                self.blue_info[host][-1] = "No"

        self.info = info

        if cardiff:
            return self._create_cardiff_vector(success)

        if self.output_mode == "table":
            return self._create_blue_table(success)
        elif self.output_mode == "anomaly":
            anomaly_obs["success"] = success
            return anomaly_obs
        elif self.output_mode == "raw":
            return observation
        elif self.output_mode == "vector":
            return self._create_blue_vector(success)
        else:
            raise NotImplementedError("Invalid output_mode for BlueTableWrapper")

    def _process_blue_action(self, success):
        action = self.get_last_action(agent="Blue")
        if action is not None:
            name = action.__class__.__name__
            hostname = (
                action.get_params()["hostname"]
                if name in ("Restore", "Remove")
                else None
            )

            if name == "Restore":
                self.blue_info[hostname][-1] = "No"
                # Update Red Access, if Red is aware of this machine
                ip = str(self.ip_map[hostname])
                if ip in self.red_info:
                    self.red_info[ip][4] = "None"

            # Old observation update for removing, superceded by moving any User machines to Unknow below
            elif name == "Remove":
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'
                # Update Red Access, if Red is aware of this machine
                ip = str(self.ip_map[hostname])
                if ip in self.red_info:
                    if self.red_info[ip][4] == "User":
                        self.red_info[ip][4] = "None"
    
    def _detect_anomalies(self, obs):
        if self.baseline is None:
            raise TypeError(
                "BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method."
            )

        anomaly_dict = {}

        for hostid, host in obs.items():
            if hostid == "success":
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if "Files" in host:
                baseline_files = host_baseline.get("Files", [])
                anomalous_files = []
                for f in host["Files"]:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies["Files"] = anomalous_files

            if "Processes" in host:
                baseline_processes = host_baseline.get("Processes", [])
                anomalous_processes = []
                for p in host["Processes"]:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies["Processes"] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict
    
    def _process_anomalies(self, anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if "Processes" in host_anomalies:
                connection_type = self._interpret_connections(
                    host_anomalies["Processes"]
                )
                info[hostid][-2] = connection_type
                if connection_type == "Exploit":
                    info[hostid][-1] = "User"
                    self.blue_info[hostid][-1] = "User"

            if "Files" in host_anomalies:
                malware = [f["Density"] >= 0.9 for f in host_anomalies["Files"]]
                if any(malware):
                    info[hostid][-1] = "Privileged"
                    self.blue_info[hostid][-1] = "Privileged"

        return info

    def _interpret_connections(self, activity: list):
        num_connections = len(activity)

        ports = set(
            [
                item["Connections"][0]["local_port"]
                for item in activity
                if "Connections" in item
            ]
        )
        port_focus = len(ports)

        remote_ports = set(
            [
                item["Connections"][0].get("remote_port")
                for item in activity
                if "Connections" in item
            ]
        )
        if None in remote_ports:
            remote_ports.remove(None)

        if num_connections >= 3 and port_focus >= 3:
            anomaly = "Scan"
        elif 4444 in remote_ports:
            anomaly = "Exploit"
        elif num_connections >= 3 and port_focus == 1:
            anomaly = "Exploit"
        elif "Service Name" in activity[0]:
            anomaly = "None"
        else:
            anomaly = "Scan"

        return anomaly

    def _create_cardiff_vector(self, success):
        table = self._create_blue_table(success)._rows

        proto_vector = []
        for row in table:
            # Activity
            activity = row[3]
            if activity == 'None':
                value = [0, 0]
            elif activity == 'Scan':
                value = [1, 0]
            elif activity == 'Exploit':
                value = [1, 1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

            # Compromised
            compromised = row[4]
            if compromised == 'No':
                value = [0, 0]
            elif compromised == 'Unknown':
                value = [1, 0]
            elif compromised == 'User':
                value = [0, 1]
            elif compromised == 'Privileged':
                value = [1, 1]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        return np.array(proto_vector)

    def _create_blue_vector(self, success):
        table = self._create_blue_table(success)._rows

        # What does the table look like?
        #for row in table:
        #    print(row)

        proto_vector = []

        # five flags for each host:
        # 1 - is this host uncompromised
        # 2 - was this host exploited last turn
        # 3 - is the status of this host unknown (has been exploited, and not restored or analyzed)
        # 4 - does Red have at least User privleges (exploited, and never removed or restored)
        # 5 - does Red have Root privleges (host was analyzed)

        # for each of the 5 machines, check these flags and add them to the observation vector
        for row in table:
            if ((row[2] == "Defender") or (row[2] == "User0")):
                continue

            # Flag 1
            compromised = row[4]
            if compromised == "No":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 2
            activity = row[3]
            if activity == "Exploit":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 3
            if ((compromised != "No") and (compromised != "Privileged")):
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 4
            if compromised == "User":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 5
            if compromised == "Privileged":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
        
        turn_vector = [0]*(self.turns_per_game+1)
        turn_vector[self.turn] = 1
        proto_vector.extend(turn_vector)

        # Final observation vector?
        # print("Final observation vector is: ", np.array(proto_vector))
        # print("Length of final obs vector is: ", len(np.array(proto_vector)))

        return np.array(proto_vector)
    
    def _create_blue_table(self, success):
        table = PrettyTable(
            ["Subnet", "IP Address", "Hostname", "Activity", "Compromised"]
        )
        for hostid in self.info:
            table.add_row(self.info[hostid])

        table.sortby = "Hostname"
        table.success = success
        return table
    
    def get_blue_table(self, output_mode="blue_table"):
        if output_mode == "blue_table":
            return self._create_blue_table(success=None)
        elif output_mode == "true_table":
            return self.env.get_table()

    def red_observation_change(self, observation, action):
        self.success = observation["success"]

        self.step_counter += 1
        if self.step_counter <= 0:
            self._initial_red_obs(observation)
        elif self.success:
            self._update_red_info(observation, action)

        if self.output_mode == "table":
            obs = self._create_red_table()
        elif self.output_mode == "vector":
            obs = self._create_red_vector()
        elif self.output_mode == "raw":
            obs = observation
        else:
            raise NotImplementedError("Invalid output_mode")
    
        return obs
    
    def _initial_red_obs(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            interface = host["Interface"][0]
            subnet = interface["Subnet"]
            self.known_subnets.add(subnet)
            ip = str(interface["IP Address"])
            hostname = host["System info"]["Hostname"]
            self.red_info[ip] = [str(subnet), str(ip), hostname, False, "Privileged"]

    def _update_red_info(self, obs, action):
        name = action.__class__.__name__
        if name == "DiscoverRemoteSystems":
            self._add_ips(obs)
        elif name == "DiscoverNetworkServices":
            ip = str(action.ip_address)
            if (ip in self.red_info):
                self.red_info[ip][3] = True
        elif name == "ExploitRemoteService":
            self._process_exploit(obs)
        elif name == "PrivilegeEscalate":
            hostname = str(action.hostname)
            if(str(self.ip_map[hostname]) in self.red_info):
                self._process_priv_esc(obs, hostname)
        elif name == "Impact":
            hostname = str(action.hostname)
            if hostname == "Op_Server0":
                server_ip = str(self.ip_map[hostname])
                if(server_ip in self.red_info):
                    access = self.red_info[server_ip][4]
                    if access == "Privileged":
                        self.impact_status = [1]

    def _add_ips(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            for interface in host["Interface"]:
                ip = interface["IP Address"]
                subnet = interface["Subnet"]
                if subnet not in self.known_subnets:
                    self.known_subnets.add(subnet)
                if str(ip) not in self.red_info:
                    subnet = self._get_subnet(ip)
                    hostname = self._generate_name("HOST")
                    self.red_info[str(ip)] = [subnet, str(ip), hostname, False, "None"]
                # elif self.red_info[str(ip)][0].startswith("UNKNOWN_"):
                # elif "UNKNOWN_" in self.red_info[str(ip)][0]:
                else:
                    self.red_info[str(ip)][0] = self._get_subnet(ip)
    
    def _get_subnet(self, ip):
        for subnet in self.known_subnets:
            if ip in subnet:
                return str(subnet)
        return self._generate_name("SUBNET")
    
    def _generate_name(self, datatype: str):
        self.id_tracker += 1
        unique_id = "UNKNOWN_" + datatype + ": " + str(self.id_tracker)
        return unique_id

    def _process_exploit(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            if "Sessions" in host:         
                ip = str(host["Interface"][0]["IP Address"])
                hostname = host["System info"]["Hostname"]
                session = host["Sessions"][0]

                # if Red already has Root access, it keeps this access and does not drop to User
                if self.red_info[ip][4] == "Privileged":
                    access = "Privileged"
                else:
                    # access = "Privileged" if "Username" in session else "User" # this needs to be changed in original CybORG code. Privileged means the username is SYSTEM or root
                    # this access is true when all machines use windows_user_host1 image (which has 'vagrant' username on a regular exploit, no username on a critical exploit,
                    # and 'SYSTEM' username on a privilege escalation)
                    # only the OpServer uses its own image (which has 'pi' username on a regular exploit, and 'root' username on a privilege escalation)
                    access = "User"
                    if "Username" in session:
                        if session['Username'] in {'root','SYSTEM'}:
                            access = "Privileged"
                    else:
                        if hostname != "Op_Host0":
                            access = "Privileged"
                        
                self.red_info[ip][2] = hostname
                self.red_info[ip][4] = access
    
    def _process_priv_esc(self, obs, hostname):
        if obs["success"] == False:
            self.red_info[str(self.ip_map[hostname])][4] = 'None'

        else:
            for hostid in obs:
                if hostid == "success":
                    continue
                host = obs[hostid]
                ip = host["Interface"][0]["IP Address"]
                if "Sessions" in host:
                    
                    access = "Privileged"
                    self.red_info[str(ip)][4] = access
                    # New: learn subnet with Root access to a host 
                    subnet = host["Interface"][0]["Subnet"]
                    if subnet not in self.known_subnets:
                        self.known_subnets.add(subnet)
                        self.red_info[str(ip)][0] = subnet
                else:
                    subnet = self._get_subnet(ip) # this line is replaced above
                    hostname = self._generate_name("HOST")

                    if str(ip) not in self.red_info:
                        self.red_info[str(ip)] = [
                            subnet,
                            str(ip),
                            hostname,
                            False,
                            "None",
                        ]
                    else:
                        self.red_info[str(ip)][0] = subnet
                        self.red_info[str(ip)][2] = hostname
    
    def _create_red_vector(self):
        table = self._create_red_table()._rows

        # What does the table look like?
        #for row in table:
        #    print(row)

        # success flag for previous action. Not included for now
        # success = int(self.success.value) if self.success.value < 2 else 0

        # Four flags for each subnet:
        # s1 - are the IPs still unknown
        # s2 - are the hosts still not scanned
        # s3 - still no User access on any machine
        # s4 - still no Root access on any machine

        # Four flags for each host:
        # h1 - has this host been scanned
        # h2/h3/h4 - is there None, User, or Privileged access on this host

        # one bit to track if Red is currently impacting

        subnet_status = [0]*2*len(self.subnets)
        host_status = [1,0,0]*len(self.hostnames)
        
        for host in self.hostnames:
            if str(self.ip_map[host]) in self.red_info:
                host_ip = str(self.ip_map[host])
                index = self.hostnames.index(host)

                # s1, are the IPs in this subnet known
                if self.subnet_scan_status[0] == 0:
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        self.subnet_scan_status[0] = 1
                elif self.subnet_scan_status[1] == 0:
                    if (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        self.subnet_scan_status[1] = 1
                    
                # h1 - has this host been scanned
                if int(self.red_info[host_ip][3]):
                    self.host_scan_status[index] = 1
                    # update s2
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        self.subnet_scan_status[2] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        self.subnet_scan_status[3] = 1

                # h2/h3/h4 - is there none/user/root access on this host
                access = self.red_info[host_ip][4]
                if access == "User":
                    host_status[(3*index)] = 0
                    host_status[(3*index)+1] = 1
                    host_status[(3*index)+2] = 0
                    # update s3
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        subnet_status[0] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        subnet_status[1] = 1

                elif access == "Privileged":
                    host_status[(3*index)] = 0
                    host_status[(3*index)+1] = 0
                    host_status[(3*index)+2] = 1
                    # update s4
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        subnet_status[2] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        subnet_status[3] = 1

                if host == "Op_Server0":
                    if access != "Privileged":    
                        self.impact_status = [0]

        proto_vector = []
        proto_vector.extend(self.host_scan_status)      # 5 bits
        proto_vector.extend(host_status)                # 15 bits
        proto_vector.extend(self.subnet_scan_status)    # 4 bits
        proto_vector.extend(subnet_status)              # 4 bits
        proto_vector.extend(self.impact_status)         # 1 bit
        turn_vector = [0]*(self.turns_per_game+1)
        turn_vector[self.turn] = 1
        proto_vector.extend(turn_vector)                # 1 bit

        # Final observation vector?
        # print("Final observation vector is: ", np.array(proto_vector))
        # print("Length of final obs vector is: ", len(np.array(proto_vector)))

        return np.array(proto_vector)
    
    def _create_red_table(self):
        # The table data is all stored inside the ip nodes
        # which form the rows of the table
        table = PrettyTable(
            [
                "Subnet",
                "IP Address",
                "Hostname",
                "Scanned",
                "Access",
            ]
        )
        for ip in self.red_info:
            table.add_row(self.red_info[ip])

        table.sortby = "IP Address"
        table.success = self.success
        return table
    
    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        if agent == "Blue" and self.output_mode == "table":
            output = self.get_table()
        else:
            output = self.get_attr("get_observation")(agent)

        return output

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")