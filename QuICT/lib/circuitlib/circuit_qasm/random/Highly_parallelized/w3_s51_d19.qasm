OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(0.49682341990298745) q[1];
rz(5.942893952390081) q[0];
rz(5.690127335517354) q[2];
rz(2.0100549669200443) q[1];
rz(0.9179429175179634) q[0];
rz(2.1828640430125885) q[2];
rz(4.813229950731905) q[1];
rz(1.1449641228123197) q[0];
rz(4.216616509738671) q[2];
rz(1.64201956280578) q[0];
rz(4.804449094867578) q[2];
rz(0.5371655023370252) q[1];
rz(4.439869546686517) q[1];
cx q[0], q[2];
rz(4.9543833790910305) q[0];
rz(1.9357006354917496) q[1];
rz(4.392173815260055) q[2];
rz(3.2478929414907807) q[1];
rz(4.598514522852583) q[2];
rz(3.2384265587063923) q[0];
rz(0.6657199215239534) q[0];
rz(1.2574868970483044) q[1];
rz(2.185448322749248) q[2];
rz(3.4958637720678034) q[1];
cx q[2], q[0];
rz(3.4729419757906865) q[1];
rz(5.845529331505569) q[0];
rz(5.359535914149823) q[2];
rz(0.1651194753090413) q[1];
rz(0.37431925224759094) q[2];
rz(0.9239286969201622) q[0];
rz(2.9852258143291457) q[0];
rz(4.669377585897683) q[1];
rz(1.0295994787008895) q[2];
rz(0.6844839833668311) q[2];
rz(4.237964469788524) q[1];
rz(4.8071561753447964) q[0];
cx q[1], q[2];
rz(2.7979402705211167) q[0];
rz(3.135209697232553) q[0];
rz(1.4136411649202578) q[1];
rz(4.177154338407055) q[2];
rz(1.6723392721941528) q[2];
rz(4.911703932319809) q[1];
rz(3.816844182245695) q[0];
rz(5.874562398646166) q[1];
cx q[2], q[0];
rz(5.044919182831364) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];