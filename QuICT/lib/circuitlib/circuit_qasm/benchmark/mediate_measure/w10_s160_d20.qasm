OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(0.580060984983041) q[4];
rz(5.609132642757331) q[9];
rz(3.5588369296118) q[6];
rz(5.3537590251828) q[3];
rz(4.931605893004794) q[7];
rz(2.557262179979875) q[5];
rz(0.9204086942888939) q[0];
cx q[1], q[8];
rz(0.1935461244167832) q[2];
rz(3.10140929473891) q[9];
rz(3.4332374147663622) q[8];
rz(2.3176013115443057) q[4];
rz(4.393805868807923) q[3];
rz(4.248617005159626) q[6];
rz(0.17295755400831359) q[2];
rz(1.2764933979838367) q[7];
rz(4.8464911157911486) q[5];
rz(5.877825129272005) q[1];
rz(3.8663786606085333) q[0];
rz(0.33203256601077774) q[6];
rz(1.9366070631563912) q[8];
rz(0.9405242773573002) q[5];
rz(3.797153549169773) q[3];
cx q[1], q[7];
rz(3.486893303651015) q[9];
rz(4.986990954181833) q[4];
rz(5.611106630448035) q[0];
rz(0.48966823878501803) q[2];
rz(0.07450261940295001) q[8];
rz(2.962362674700899) q[6];
rz(5.804099126768326) q[9];
rz(3.230028158045318) q[2];
rz(2.0517942099340676) q[3];
rz(2.527261614331244) q[7];
cx q[4], q[1];
rz(1.5319993231293263) q[5];
rz(2.519430487425129) q[0];
rz(2.7926967766518067) q[0];
cx q[5], q[8];
cx q[9], q[3];
cx q[1], q[6];
rz(1.494474303572495) q[7];
rz(0.6941318913656733) q[4];
rz(5.885040852673197) q[2];
cx q[4], q[9];
rz(0.9271585427777654) q[3];
rz(5.9454764065522605) q[7];
cx q[2], q[8];
rz(0.06278599668562194) q[1];
rz(0.9813379160066971) q[5];
rz(4.531513672142517) q[0];
rz(0.9582539422707117) q[6];
rz(2.390645347547958) q[2];
cx q[7], q[0];
rz(4.347682714597854) q[1];
cx q[9], q[3];
cx q[5], q[8];
rz(1.053861894109129) q[6];
rz(0.7037943805639651) q[4];
rz(6.056368675539127) q[0];
rz(3.1358739168477907) q[6];
cx q[5], q[1];
rz(6.2496489786651415) q[4];
cx q[7], q[8];
cx q[9], q[3];
rz(4.5360726703772) q[2];
rz(5.903514660858663) q[0];
rz(1.0892794302915816) q[5];
rz(5.806514936227479) q[2];
rz(4.585515289138677) q[3];
cx q[6], q[1];
rz(4.645953022108224) q[7];
rz(3.0231721413029025) q[8];
rz(3.934611764353016) q[4];
rz(1.896010334697962) q[9];
cx q[8], q[1];
rz(1.1171624609329156) q[7];
rz(5.174973724791752) q[5];
rz(5.235561454958291) q[9];
rz(0.8646276560440753) q[2];
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
rz(5.1327526258754155) q[4];
rz(3.6341573537888583) q[0];
rz(4.7152999113080085) q[6];
rz(1.7931775119551263) q[3];
rz(5.456809585506011) q[1];
rz(1.6992783148535884) q[5];
cx q[4], q[6];
rz(1.0816854959141282) q[3];
cx q[8], q[0];
rz(3.9488967939337267) q[9];
rz(0.4346814305478907) q[2];
rz(1.357589214638221) q[7];
rz(4.372004128374209) q[2];
cx q[6], q[9];
rz(4.499600685608361) q[4];
rz(6.013110182882582) q[8];
cx q[7], q[5];
rz(0.3975554020612222) q[3];
rz(4.977397016988371) q[0];
rz(4.404996825640144) q[1];
cx q[9], q[3];
rz(4.114905899808404) q[1];
cx q[5], q[8];
rz(0.881629192100514) q[7];
rz(2.841998472784507) q[4];
cx q[6], q[2];
rz(1.0007623553627323) q[0];
rz(1.9871815208185186) q[5];
rz(4.755902328251064) q[2];
rz(3.7365204276845634) q[4];
cx q[8], q[3];
rz(2.7263866456983377) q[1];
cx q[0], q[6];
rz(1.0367296565256805) q[7];
rz(1.275294967893657) q[9];
rz(0.6203234295454869) q[6];
rz(4.5222088805536425) q[0];
rz(4.949868386485889) q[4];
rz(3.4271909564187526) q[7];
rz(4.1632247948182615) q[1];
cx q[8], q[9];
rz(2.88720145188102) q[5];
cx q[2], q[3];
rz(3.2855108862147238) q[7];
rz(5.025056201763983) q[3];
rz(0.3451759101591267) q[6];
rz(4.783498893723266) q[8];
rz(0.9575125538248592) q[4];
rz(2.635727013449308) q[9];
cx q[0], q[2];
cx q[5], q[1];
rz(0.22216800674673476) q[1];
cx q[6], q[9];
rz(3.913591927849423) q[3];
rz(5.562737104388134) q[4];
rz(5.486205649317541) q[2];
rz(4.593634735851449) q[7];
cx q[8], q[0];
rz(0.9155275539579156) q[5];
rz(3.9306892880363553) q[4];
rz(6.196177896643192) q[1];
rz(1.8432606526963458) q[7];
rz(0.4774704557980743) q[0];
rz(5.565964793778262) q[2];
rz(3.7103954823958567) q[8];
rz(5.746229237324492) q[5];
rz(1.5006234215039) q[9];
rz(4.844295055324149) q[3];
rz(1.0020553915013652) q[6];
rz(1.5540448237156306) q[9];