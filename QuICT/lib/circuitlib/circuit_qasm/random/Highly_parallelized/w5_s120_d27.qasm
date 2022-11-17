OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[3], q[0];
rz(5.469417786372718) q[4];
rz(1.4667142633264019) q[1];
rz(0.05307506659010518) q[2];
rz(2.359881795024574) q[3];
rz(0.6666637638862659) q[4];
rz(3.72417026710139) q[0];
rz(5.740301713442427) q[1];
rz(5.87702971176661) q[2];
rz(4.55773077654984) q[2];
rz(1.4470795477699636) q[4];
rz(5.976620478156294) q[1];
rz(0.45851911550363683) q[0];
rz(1.0303956449962313) q[3];
rz(2.95602499470619) q[0];
rz(0.5358603379151605) q[1];
cx q[3], q[2];
rz(2.215714199065683) q[4];
rz(3.172842174023451) q[1];
rz(2.5192266159685603) q[2];
rz(1.9637624620814595) q[3];
cx q[4], q[0];
rz(4.391459546298424) q[1];
rz(4.892155314317995) q[3];
rz(1.0548487594451037) q[4];
cx q[2], q[0];
rz(4.30358613863692) q[3];
rz(0.38261530317606574) q[2];
rz(1.0465809380321611) q[4];
cx q[1], q[0];
rz(1.1996685121641657) q[4];
rz(2.020895163246624) q[2];
rz(1.1913602915912374) q[3];
rz(4.58789064483125) q[1];
rz(4.855780232058517) q[0];
cx q[3], q[1];
rz(6.134547763433414) q[0];
rz(0.3210483471810749) q[2];
rz(5.604080739817443) q[4];
rz(0.7604732956292489) q[2];
rz(5.033064968816472) q[4];
rz(2.5999077030077817) q[1];
rz(1.4620450004048655) q[0];
rz(2.956525255783499) q[3];
rz(3.2057931615469912) q[4];
rz(4.735607842261341) q[2];
rz(0.9855013462805621) q[3];
rz(1.7695501932428086) q[0];
rz(4.104149430305794) q[1];
rz(4.058757481347787) q[3];
rz(4.485858397157143) q[2];
cx q[0], q[1];
rz(1.565082012379975) q[4];
rz(1.5245690027837497) q[0];
rz(0.3647210107644008) q[4];
rz(1.9895945821331071) q[1];
rz(2.4758016523251745) q[3];
rz(4.192050869775286) q[2];
rz(3.993296141540495) q[4];
rz(4.6472497376501) q[0];
rz(5.3710679704067) q[3];
rz(1.3013308681646407) q[2];
rz(5.766306995045868) q[1];
rz(5.7776879714939655) q[4];
rz(2.828653373922394) q[0];
rz(2.826757897738195) q[1];
rz(3.192728831880243) q[3];
rz(0.8248927487245918) q[2];
rz(3.7798374939929418) q[3];
rz(4.73330173418463) q[4];
rz(3.242346269105358) q[2];
rz(4.7140810767548516) q[1];
rz(3.4162999037882202) q[0];
rz(5.234852322266314) q[2];
rz(0.08307662706241892) q[1];
rz(2.611710845971476) q[0];
rz(3.522195770730772) q[3];
rz(1.2813377069471175) q[4];
rz(5.774794577384986) q[0];
rz(1.0191511295754128) q[4];
rz(2.2967475233221117) q[3];
rz(1.5165413478813388) q[1];
rz(5.77340088857962) q[2];
rz(6.051349818563456) q[1];
cx q[3], q[0];
rz(5.958907132002355) q[4];
rz(6.129782052853835) q[2];
rz(1.7520255561106555) q[2];
cx q[3], q[4];
rz(1.6702366497562522) q[1];
rz(1.5656196121783177) q[0];
rz(0.10821490712591378) q[1];
cx q[4], q[0];
rz(3.7807702612624285) q[2];
rz(2.8520320469693994) q[3];
cx q[3], q[4];
rz(0.5119474856857507) q[1];
rz(5.7497562395392805) q[0];
rz(5.4148635100919185) q[2];
rz(1.4621463719500911) q[4];
rz(4.325400722348792) q[0];
rz(4.816449836471085) q[2];
rz(3.8982279802505078) q[1];
rz(3.4849195177789447) q[3];
rz(0.4392281950991461) q[2];
cx q[0], q[1];
rz(2.806992418065926) q[3];
rz(5.461875535333977) q[4];
rz(0.7635549740272187) q[1];
rz(2.0367948317395217) q[4];
rz(5.684724448276116) q[3];
rz(4.314360571553132) q[2];
rz(0.7856538491798124) q[0];
cx q[0], q[4];
rz(5.456690769969049) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];