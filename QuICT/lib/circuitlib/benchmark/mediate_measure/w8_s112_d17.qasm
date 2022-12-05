OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
cx q[2], q[4];
rz(4.657302878163667) q[7];
rz(4.8788220990644415) q[6];
rz(6.1934629974805935) q[0];
rz(5.882581090485247) q[1];
cx q[3], q[5];
rz(0.4723190372579709) q[0];
cx q[6], q[7];
rz(1.5142175225439722) q[2];
rz(3.008254664035474) q[5];
rz(4.304642414542749) q[4];
rz(3.0861923102952082) q[3];
rz(3.7741227317278434) q[1];
rz(1.478619208559287) q[5];
rz(2.6449541915839196) q[1];
rz(4.333872792145957) q[0];
rz(5.890730503884419) q[3];
cx q[4], q[2];
cx q[7], q[6];
rz(6.057917338208478) q[2];
cx q[3], q[7];
rz(6.0986462115502125) q[6];
rz(3.368358275075782) q[1];
rz(2.6314708876909014) q[0];
rz(1.3245204492033638) q[4];
rz(0.37095215555020633) q[5];
cx q[7], q[2];
cx q[3], q[5];
rz(0.9373459621040593) q[0];
rz(0.37862417151625777) q[6];
rz(4.101169933169926) q[1];
rz(2.6706000201357702) q[4];
rz(4.042833019363374) q[4];
rz(0.3832045597017186) q[5];
rz(1.5969191506323177) q[6];
rz(3.7938836783610435) q[3];
rz(5.252608221386028) q[7];
rz(4.304800701040421) q[2];
rz(2.493254378862398) q[1];
rz(0.6940165359121658) q[0];
rz(2.4265150156090955) q[3];
rz(0.20617877941158438) q[7];
rz(3.971037762774806) q[5];
rz(5.929188498010929) q[2];
cx q[0], q[4];
rz(5.274571458265398) q[6];
rz(0.05855989217608995) q[1];
rz(1.6780373099900348) q[2];
rz(4.126590446081472) q[3];
rz(5.036728081736793) q[6];
rz(5.076768474736375) q[5];
rz(5.389648760244276) q[0];
rz(5.231060256274358) q[1];
rz(3.8934156041925485) q[4];
rz(4.475026958747549) q[7];
rz(4.766403990179005) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
rz(2.055391258360206) q[7];
rz(0.8010885881103961) q[3];
rz(4.855671924181551) q[1];
rz(1.906908990020853) q[2];
rz(2.9408700553626668) q[4];
rz(4.270545879729521) q[6];
rz(3.4714971575760423) q[5];
rz(0.003190811544257431) q[1];
rz(2.178736705544364) q[0];
rz(1.5861409414021777) q[6];
cx q[3], q[5];
rz(4.565813409058218) q[7];
cx q[2], q[4];
cx q[2], q[3];
rz(1.8960731345419637) q[4];
cx q[1], q[5];
rz(5.817632699134901) q[7];
cx q[0], q[6];
rz(5.523986965790944) q[3];
cx q[5], q[4];
rz(4.341908673415082) q[1];
rz(2.9911354405486055) q[2];
rz(4.002041679716918) q[7];
cx q[6], q[0];
rz(3.6969552283511224) q[5];
rz(3.8984027214589707) q[2];
rz(3.3141481270591946) q[6];
rz(2.44072268251782) q[1];
rz(0.18893405034916025) q[3];
cx q[7], q[4];
rz(0.9804499501814614) q[0];
cx q[1], q[4];
cx q[7], q[6];
rz(1.766897839207504) q[5];
rz(3.612417266952364) q[2];
cx q[3], q[0];
cx q[4], q[5];
rz(0.2851712472364771) q[1];
rz(0.9718757192875314) q[0];
rz(1.2253287848435725) q[6];
rz(5.167837746889953) q[3];
rz(1.74102859972029) q[2];
rz(5.512508430091171) q[7];
rz(5.968249287712164) q[5];
rz(5.586185201205513) q[4];
rz(2.334859754893623) q[7];
rz(4.779104503818649) q[2];
rz(5.073357127423569) q[0];