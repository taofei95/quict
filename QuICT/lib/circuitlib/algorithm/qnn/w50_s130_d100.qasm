OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
x q[1];
x q[4];
x q[6];
x q[11];
x q[12];
x q[13];
x q[16];
x q[20];
x q[21];
x q[22];
x q[23];
x q[24];
x q[25];
x q[28];
x q[29];
x q[30];
x q[31];
x q[32];
x q[33];
x q[34];
x q[36];
x q[39];
x q[40];
x q[42];
x q[43];
x q[44];
x q[46];
x q[47];
x q[48];
x q[0];
h q[0];
ryy(0.7487459778785706) q[0], q[49];
ryy(0.593354344367981) q[1], q[49];
ryy(0.9271402359008789) q[2], q[49];
ryy(0.5678484439849854) q[3], q[49];
ryy(0.9490346312522888) q[4], q[49];
ryy(0.5511136054992676) q[5], q[49];
ryy(0.56519615650177) q[6], q[49];
ryy(0.808776319026947) q[7], q[49];
ryy(0.8113332390785217) q[8], q[49];
ryy(0.08036714792251587) q[9], q[49];
ryy(0.6802666783332825) q[10], q[49];
ryy(0.5713329315185547) q[11], q[49];
ryy(0.27375227212905884) q[12], q[49];
ryy(0.1123809814453125) q[13], q[49];
ryy(0.467316210269928) q[14], q[49];
ryy(0.957653820514679) q[15], q[49];
ryy(0.5085909962654114) q[16], q[49];
ryy(0.9745979309082031) q[17], q[49];
ryy(0.29287880659103394) q[18], q[49];
ryy(0.23466557264328003) q[19], q[49];
ryy(0.45443058013916016) q[20], q[49];
ryy(0.9834882616996765) q[21], q[49];
ryy(0.7500327229499817) q[22], q[49];
ryy(0.3123946189880371) q[23], q[49];
ryy(0.8573411703109741) q[24], q[49];
ryy(0.8430613279342651) q[25], q[49];
ryy(0.430436909198761) q[26], q[49];
ryy(0.8753570318222046) q[27], q[49];
ryy(0.5116057395935059) q[28], q[49];
ryy(0.16779768466949463) q[29], q[49];
ryy(0.07111907005310059) q[30], q[49];
ryy(0.36746037006378174) q[31], q[49];
ryy(0.9544661641120911) q[32], q[49];
ryy(0.9419788718223572) q[33], q[49];
ryy(0.584007740020752) q[34], q[49];
ryy(0.2029639482498169) q[35], q[49];
ryy(0.1621832251548767) q[36], q[49];
ryy(0.38205188512802124) q[37], q[49];
ryy(0.1640048623085022) q[38], q[49];
ryy(0.16617631912231445) q[39], q[49];
ryy(0.05700463056564331) q[40], q[49];
ryy(0.4764564633369446) q[41], q[49];
ryy(0.5125789046287537) q[42], q[49];
ryy(0.4859732985496521) q[43], q[49];
ryy(0.36010587215423584) q[44], q[49];
ryy(0.6276898384094238) q[45], q[49];
ryy(0.1429438591003418) q[46], q[49];
ryy(0.27786940336227417) q[47], q[49];
ryy(0.4240495562553406) q[48], q[49];
rzz(0.5544843673706055) q[0], q[49];
rzz(0.2741280794143677) q[1], q[49];
rzz(0.03531813621520996) q[2], q[49];
rzz(0.17505556344985962) q[3], q[49];
rzz(0.001012563705444336) q[4], q[49];
rzz(0.020992696285247803) q[5], q[49];
rzz(0.5144299268722534) q[6], q[49];
rzz(0.12410622835159302) q[7], q[49];
rzz(0.43382561206817627) q[8], q[49];
rzz(0.883322536945343) q[9], q[49];
rzz(0.771997332572937) q[10], q[49];
rzz(0.9558854699134827) q[11], q[49];
rzz(0.009471356868743896) q[12], q[49];
rzz(0.8474140763282776) q[13], q[49];
rzz(0.07505875825881958) q[14], q[49];
rzz(0.3187897205352783) q[15], q[49];
rzz(0.25872504711151123) q[16], q[49];
rzz(0.3838634490966797) q[17], q[49];
rzz(0.7219364643096924) q[18], q[49];
rzz(0.4493032693862915) q[19], q[49];
rzz(0.15068358182907104) q[20], q[49];
rzz(0.1413966417312622) q[21], q[49];
rzz(0.10797899961471558) q[22], q[49];
rzz(0.6412928104400635) q[23], q[49];
rzz(0.8990469574928284) q[24], q[49];
rzz(0.4256440997123718) q[25], q[49];
rzz(0.42373377084732056) q[26], q[49];
rzz(0.23241013288497925) q[27], q[49];
rzz(0.6488726735115051) q[28], q[49];
rzz(0.7429214119911194) q[29], q[49];
rzz(0.13503092527389526) q[30], q[49];
rzz(0.8266542553901672) q[31], q[49];
rzz(0.13463270664215088) q[32], q[49];
rzz(0.9812679290771484) q[33], q[49];
rzz(0.020362555980682373) q[34], q[49];
rzz(0.361264169216156) q[35], q[49];
rzz(0.4366946220397949) q[36], q[49];
rzz(0.7457587122917175) q[37], q[49];
rzz(0.8816680908203125) q[38], q[49];
rzz(0.9916888475418091) q[39], q[49];
rzz(0.8924391865730286) q[40], q[49];
rzz(0.6938431859016418) q[41], q[49];
rzz(0.16521161794662476) q[42], q[49];
rzz(0.14380806684494019) q[43], q[49];
rzz(0.4934791922569275) q[44], q[49];
rzz(0.8461458683013916) q[45], q[49];
rzz(0.9317045211791992) q[46], q[49];
rzz(0.6383146047592163) q[47], q[49];
rzz(0.39920753240585327) q[48], q[49];
h q[0];