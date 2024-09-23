import itertools

android = [
    {
        "log_data": "03-17 16:13:38.859  2227  2227 D TextView: visible is system.time.showampm",
        "response": '["Date", "Time", "Pid", "Tid", "Level", "Component", "Content"]'
    },
    {
        "log_data": '03-17 16:13:38.819  1702  8671 D PowerManagerService: acquire lock=233570404, flags=0x1, tag="View Lock", name=com.android.systemui, ws=null, uid=10037, pid=2227',
        "response": '["Date", "Time", "Pid", "Tid", "Level", "Component", "Content"]'
    }
]

apache = [
    {
        "log_data": "[Sun Dec 04 04:47:44 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties",
        "response": '["Time", "Level", "Content"]'
    },
    {
        "log_data": "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6",
        "response": '["Time", "Level", "Content"]'
    }
]


bgl = [
    {
        "log_data": "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
        "response": '["Label", "Timestamp", "Date", "Node", "NodeRepeat", "Type", "Component", "Level", "Content"]'
    },
    {
        "log_data": "APPREAD 1117869876 2005.06.04 R27-M1-N4-I:J18-U01 2005-06-04-00.24.36.222560 R27-M1-N4-I:J18-U01 RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:33370",
        "response": '["Label", "Timestamp", "Date", "Node", "NodeRepeat", "Type", "Component", "Level", "Content"]'
    }
]

hadoop = [
    {
        "log_data": "2015-10-18 18:01:47,978 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster for application appattempt_1445144423722_0020_000001",
        "response": '["Date", "Time", "Level", "Process", "Component", "Content"]'
    },
    {
        "log_data": "2015-10-18 18:01:53,869 INFO [AsyncDispatcher event handler] org.apache.hadoop.mapreduce.v2.app.job.impl.TaskImpl: task_1445144423722_0020_m_000000 Task Transitioned from NEW to SCHEDULED",
        "response": '["Date", "Time", "Level", "Process", "Component", "Content"]'
    }
]

hdfs = [
    {
        "log_data": "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
        "response": '["Date", "Time", "Pid", "Level", "Component", "Content"]'
    },
    {
        "log_data": "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864",
        "response": '["Date", "Time", "Pid", "Level", "Component", "Content"]'
    }
]

healthapp = [
    {
        "log_data": "20171223-22:15:29:606|Step_LSC|30002312|onStandStepChanged 3579",
        "response": '["Time", "Component", "Pid", "Content"]'
    },
    {
        "log_data": "20171223-22:15:35:23|Step_StandReportReceiver|30002312|screen status unknown,think screen on",
        "response": '["Time", "Component", "Pid", "Content"]'
    },
]

hpc = [
    {
        "log_data": "350766 node-109 unix.hw state_change.unavailable 1084680778 1 Component State Change: Component \042alt0\042 is in the unavailable state (HWID=3180)",
        "response": '["LogId", "Node", "Component", "State", "Time", "Flag", "Content"]'
    },
    {
        "log_data": "2568643 node-70 action start 1074119817 1 clusterAddMember  (command 1902)",
        "response": '["LogId", "Node", "Component", "State", "Time", "Flag", "Content"]'
    }
]

linux = [
    {
        "log_data": "Jun 15 02:04:59 combo sshd(pam_unix)[20882]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root",
        "response": '["Month", "Date", "Time", "Level", "Component", "PID", "Content"]'
    },
    {
        "log_data": "Jun 17 07:07:00 combo ftpd[29504]: connection from 24.54.76.216 (24-54-76-216.bflony.adelphia.net) at Fri Jun 17 07:07:00 2005",
        "response": '["Month", "Date", "Time", "Level", "Component", "PID", "Content"]'
    }
]

mac = [
    {
        "log_data": "Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: IOThunderboltSwitch<0>(0x0)::listenerCallback - Thunderbolt HPD packet for route = 0x0 port = 11 unplug = 0",
        "response": '["Month", "Date", "Time", "User", "Component", "PID", "Address", "Content"]'
    },
    {
        "log_data": "Jul  1 09:01:05 calvisitor-10-105-160-95 com.apple.CDScheduler[43]: Thermal pressure state: 1 Memory pressure state: 0",
        "response": '["Month", "Date", "Time", "User", "Component", "PID", "Address", "Content"]'
    }    
]

openssh = [
    {
        "log_data": "Dec 10 06:55:46 LabSZ sshd[24200]: reverse mapping checking getaddrinfo for ns.marryaldkfaczcz.com [173.234.31.186] failed - POSSIBLE BREAK-IN ATTEMPT!",
        "response": '["Month", "Day", "Time", "Component", "Pid", "Content"]'
    },
    {
        "log_data": "Dec 10 07:07:38 LabSZ sshd[24206]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=ec2-52-80-34-196.cn-north-1.compute.amazonaws.com.cn",
        "response": '["Month", "Day", "Time", "Component", "Pid", "Content"]'
    }    
]

openstack = [
    {
        "log_data": 'nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-2096-447d-96ea-a692162415ae 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2477829',
        "response": '["Logrecord", "Date", "Time", "Pid", "Level", "Component", "ADDR", "Content"]'
    },
    {
        "log_data": 'nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:00:04.693 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] During sync_power_state the instance has a pending task (spawning). Skip.',
        "response": '["Logrecord", "Date", "Time", "Pid", "Level", "Component", "ADDR", "Content"]'
    }    
]

proxifier = [
    {
        "log_data": "[10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS",
        "response": '["Time", "Program", "Link", "Content"]'
    },
    {
        "log_data": "[10.30 17:02:17] putty.exe - 183.62.156.108:22 open through proxy socks.cse.cuhk.edu.hk:5070 SOCKS5",
        "response": '["Time", "Program", "Link", "Content"]'
    },
]

spark = [
    {
        "log_data": "17/06/09 20:10:40 INFO executor.CoarseGrainedExecutorBackend: Registered signal handlers for [TERM, HUP, INT]",
        "response": '["Date", "Time", "Level", "Component", "Content"]'
    },
    {
        "log_data": "17/06/09 20:10:41 INFO Remoting: Remoting started; listening on addresses :[akka.tcp://sparkExecutorActorSystem@mesos-slave-07:55904]",
        "response": '["Date", "Time", "Level", "Component", "Content"]'
    }
]

thunderbird = [
    {
        "log_data": "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 crond(pam_unix)[2915]: session opened for user root by (uid=0)",
        "response": '["Label", "Timestamp", "Date", "User", "Month", "Day", "Time", "Location", "Component", "PID", "Content"]'
    },
    {
        "log_data": "- 1131566461 2005.11.09 tbird-admin1 Nov 9 12:01:01 local@tbird-admin1 /apps/x86_64/system/ganglia-3.0.1/sbin/gmetad[1682]: data_thread() got not answer from any [Thunderbird_A8] datasource",
        "response": '["Label", "Timestamp", "Date", "User", "Month", "Day", "Time", "Location", "Component", "PID", "Content"]'
    }
]

windows = [
    {
        "log_data": "2016-09-28 04:30:30, Info                  CBS    Loaded Servicing Stack v6.1.7601.23505 with Core: C:\Windows\winsxs\amd64_microsoft-windows-servicingstack_31bf3856ad364e35_6.1.7601.23505_none_681aa442f6fed7f0\cbscore.dll",
        "response": '["Date", "Time", "Level", "Component", "Content"]'
    },
    {
        "log_data": "2016-09-28 04:30:31, Info                  CBS    Warning: Unrecognized packageExtended attribute.",
        "response": '["Date", "Time", "Level", "Component", "Content"]'
    }   
]

zookeeper = [
    {
        "log_data": "2015-07-29 17:41:44,747 - INFO  [QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:2181:FastLeaderElection@774] - Notification time out: 3200",
        "response": '["Date", "Time", "Level"]'
    },
    {
        "log_data": "2015-07-29 19:04:29,071 - WARN  [SendWorker:188978561024:QuorumCnxManager$SendWorker@688] - Send worker leaving thread",
        "response": '["Date", "Time", "Level"]'
    }
]


examples = list(itertools.chain(android, 
                                apache, 
                                bgl,
                                hadoop,
                                hdfs,
                                healthapp,
                                hpc,
                                linux,
                                mac,
                                openssh,
                                openstack,
                                proxifier,
                                spark,
                                thunderbird,
                                windows,
                                zookeeper))

