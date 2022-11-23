OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
cx q[0], q[2];
rz(4.19364980970479) q[8];
rz(5.332515087880868) q[3];
rz(1.554319072900141) q[4];
rz(3.2859935998077408) q[1];
cx q[9], q[6];
cx q[5], q[7];
rz(1.6606314228302543) q[8];
rz(4.47955610441397) q[5];
cx q[0], q[4];
rz(4.634619870246862) q[3];
cx q[7], q[9];
rz(4.7847304507904695) q[2];
rz(3.2812056533210745) q[1];
rz(3.9276583487226886) q[6];
rz(5.889574670518079) q[5];
cx q[1], q[3];
rz(4.180631774585029) q[2];
cx q[4], q[6];
rz(2.173458529632282) q[7];
rz(2.9312880153137035) q[9];
cx q[0], q[8];
rz(0.4273343212279871) q[8];
rz(6.0756631265757814) q[3];
rz(2.3350990449569013) q[5];
rz(2.4125528701853853) q[7];
rz(3.747591515992915) q[4];
rz(3.800420001870094) q[2];
rz(2.0094574081603276) q[1];
cx q[6], q[9];
rz(4.64240661899798) q[0];
rz(2.8766799104554255) q[0];
rz(5.74330659800825) q[3];
rz(5.193979134883837) q[4];
rz(5.107401569375488) q[7];
rz(6.006931905865486) q[8];
cx q[2], q[9];
rz(5.159379518032391) q[1];
rz(2.6563025471868573) q[6];
rz(5.073772731687307) q[5];
rz(0.21956591450737808) q[4];
rz(2.134976157043494) q[3];
rz(4.6604202308741876) q[6];
rz(0.7135372042252732) q[5];
rz(3.6788955956033136) q[9];
rz(2.6733631756451466) q[2];
rz(2.8931454959012632) q[0];
rz(3.766190444699359) q[7];
rz(2.5230454925933916) q[8];
rz(2.366270883306576) q[1];
cx q[6], q[9];
rz(2.1815748843493274) q[7];
rz(4.888921842215031) q[5];
rz(5.432280917756117) q[8];
rz(1.69347431060248) q[4];
rz(4.107812348189098) q[3];
rz(3.8888017682688147) q[1];
cx q[2], q[0];
rz(5.502500095627651) q[3];
rz(0.16148073622048675) q[5];
rz(3.254794657577773) q[2];
rz(1.4181330792711677) q[6];
rz(5.638789206642231) q[8];
rz(3.10958303357719) q[9];
rz(5.805056149205549) q[0];
rz(5.034077661353805) q[1];
rz(4.006581368222301) q[7];
rz(5.182934784409842) q[4];
rz(3.1817299530840017) q[1];
rz(1.911440966318227) q[8];
rz(1.671982799557407) q[7];
rz(4.734247526609088) q[6];
rz(0.1371808066636914) q[4];
rz(2.7919518021199994) q[2];
cx q[5], q[0];
rz(0.8161582778628418) q[9];
rz(5.927473076478468) q[3];
cx q[8], q[9];
rz(5.367907454640197) q[0];
rz(5.535357419970863) q[5];
rz(1.2242688166285545) q[1];
rz(0.6528663115314244) q[2];
rz(0.19446672631845344) q[4];
rz(5.792121498808222) q[3];
rz(4.497462153613305) q[7];
rz(5.917305530339226) q[6];
cx q[7], q[8];
rz(1.602225093947219) q[5];
rz(0.161448990054295) q[3];
rz(1.3409632298967527) q[4];
cx q[6], q[1];
rz(3.2348473583687145) q[9];
cx q[2], q[0];
cx q[7], q[3];
rz(2.020636998073376) q[1];
rz(1.5632806537264814) q[2];
rz(3.4190087514839926) q[6];
rz(4.694637148486971) q[4];
rz(5.2660724766265705) q[0];
rz(4.866102176379496) q[8];
rz(1.1689605785757065) q[9];
rz(1.5981808474832893) q[5];
cx q[4], q[2];
rz(2.9182172115255893) q[1];
rz(3.7298194083156284) q[8];
cx q[6], q[0];
rz(5.252306375721659) q[7];
rz(0.2583800870467427) q[5];
rz(5.88977612863647) q[3];
rz(2.7388810020714005) q[9];
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