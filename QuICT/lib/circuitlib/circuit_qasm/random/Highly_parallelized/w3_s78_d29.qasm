OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(0.0781096031271967) q[1];
rz(0.09112374182406849) q[0];
rz(0.7757026462605425) q[2];
rz(1.9385060538479377) q[2];
rz(3.2486562593859625) q[0];
rz(5.601131620809075) q[1];
rz(5.1725943183919565) q[0];
rz(3.626921431991847) q[1];
rz(2.9723901393308827) q[2];
cx q[1], q[2];
rz(2.0353815635530266) q[0];
rz(0.993301603537652) q[1];
rz(3.033715906761858) q[2];
rz(0.5292961514419792) q[0];
rz(0.7200449202740307) q[2];
rz(4.729942076514437) q[1];
rz(5.002312138496303) q[0];
rz(0.9402764704202476) q[1];
rz(5.972713989024236) q[0];
rz(5.573092579711784) q[2];
rz(5.09671726508278) q[1];
rz(3.3489164019709454) q[0];
rz(2.112868929933417) q[2];
rz(2.997611132261721) q[1];
rz(1.5707787799263637) q[2];
rz(3.398351752302464) q[0];
rz(0.33128573431749947) q[0];
rz(0.5299881366978002) q[2];
rz(0.65801215813166) q[1];
cx q[0], q[2];
rz(3.7440561070938556) q[1];
rz(4.805263354809393) q[0];
rz(5.066842041287017) q[2];
rz(2.12932169903784) q[1];
rz(1.4894780031690362) q[0];
rz(1.080120375587799) q[2];
rz(4.441747410975661) q[1];
rz(2.7716187922364375) q[0];
rz(4.241188468330639) q[1];
rz(1.5674330784562407) q[2];
rz(1.7884665384761678) q[1];
rz(0.818452350029839) q[2];
rz(4.987932575672163) q[0];
rz(2.6647687021680704) q[0];
rz(2.2546664728923647) q[2];
rz(3.5086427885392037) q[1];
cx q[2], q[0];
rz(1.8000065565047694) q[1];
rz(5.464526407972322) q[0];
rz(2.682391059374587) q[1];
rz(6.020131950578894) q[2];
cx q[2], q[0];
rz(2.4988776059047035) q[1];
cx q[1], q[0];
rz(0.5454353175086233) q[2];
cx q[2], q[1];
rz(1.2384742367726067) q[0];
rz(1.659375354198575) q[0];
rz(4.940030503235575) q[1];
rz(3.4906456446505607) q[2];
rz(1.9228558834423706) q[0];
rz(2.2168551238536685) q[2];
rz(3.3554338954997753) q[1];
rz(5.887018921321003) q[2];
rz(2.5684834731265003) q[1];
rz(1.6154867183885795) q[0];
rz(4.927016210654189) q[0];
cx q[1], q[2];
rz(5.976104068789621) q[0];
cx q[2], q[1];
cx q[0], q[2];
rz(1.3628188594633786) q[1];
rz(1.5305535988187744) q[2];
rz(6.007503463402052) q[0];
rz(4.485962238817131) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];