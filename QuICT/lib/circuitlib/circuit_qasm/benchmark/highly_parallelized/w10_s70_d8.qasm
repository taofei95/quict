OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(0.8916266579639145) q[8];
rz(4.921992175513613) q[0];
rz(4.8789031495455175) q[1];
rz(5.77776858306995) q[3];
rz(3.1920391404681188) q[2];
rz(2.195898281098896) q[9];
rz(2.7977684480907294) q[6];
rz(2.0851868369826576) q[5];
cx q[4], q[7];
rz(6.064105209739836) q[9];
rz(1.8095236676396376) q[2];
rz(2.9882088078019966) q[3];
rz(1.7127813057836487) q[4];
cx q[5], q[7];
rz(4.337810604764713) q[0];
rz(1.5992644729980967) q[1];
rz(3.012936194455351) q[6];
rz(1.1201113462899985) q[8];
rz(5.336856524775886) q[4];
rz(6.10109344997512) q[0];
rz(5.210590667977895) q[6];
rz(2.3902901003686887) q[3];
rz(4.777175252775907) q[9];
rz(0.32798495691781276) q[1];
rz(2.228052982022447) q[5];
rz(4.180614070240642) q[7];
rz(1.9023233244218962) q[2];
rz(5.126547380898631) q[8];
rz(3.9094857098818587) q[7];
rz(3.030167608222986) q[2];
rz(6.111196585734185) q[0];
rz(2.32928767252534) q[5];
rz(0.24031805586015414) q[9];
rz(3.995059001197022) q[3];
rz(6.166404743275652) q[1];
rz(5.013364380104918) q[6];
rz(6.251290436669539) q[4];
rz(2.6479101200492536) q[8];
rz(6.155921042528838) q[9];
cx q[7], q[4];
rz(4.852865480642591) q[5];
rz(3.760086554319476) q[0];
rz(1.4651131686929308) q[6];
rz(4.975452439493701) q[2];
cx q[8], q[1];
rz(0.6255119391178665) q[3];
rz(4.794019794132405) q[6];
rz(5.922617442356491) q[3];
rz(2.2972099728049318) q[9];
rz(3.5711547839545146) q[8];
rz(5.487155438715366) q[2];
rz(1.031335693526358) q[1];
cx q[4], q[0];
rz(5.6958590176824435) q[7];
rz(1.9819565470213856) q[5];
cx q[9], q[6];
rz(4.597058769917568) q[3];
rz(3.267741183961388) q[1];
rz(6.005866640879699) q[0];
rz(3.7322917957741586) q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];