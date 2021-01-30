# Models

Results are on karpathy test split, beam size 5. The evaluated models are the checkpoint with the highest CIDEr on validation set. Without notice, the numbers shown are not selected. The scores are just used to verify if you are getting things right. If the scores you get is close to the number I give (it could be higher or lower), then it's ok.

# Trained with Resnet101 feature:

Collection: [link](https://drive.google.com/open?id=0B7fNdx_jAqhtcXp0aFlWSnJmb0k)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Note</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/fc.yml">FC</a></td>
<td align="center">0.953</td>
<td align="center">0.1787</td>
<td align="center"><a href="https://drive.google.com/open?id=1AG8Tulna7gan6OgmYul0QhxONDBGcdun">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
 <tr><td align="left"><a href="configs/fc_rl.yml">FC<br>+self_critical</a></td>
<td align="center">1.045</td>
<td align="center">0.1838</td>
<td align="center"><a href="https://drive.google.com/open?id=1MA-9ByDNPXis2jKG0K0Z-cF_yZz7znBc">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
 <tr><td align="left"><a href="configs/fc_nsc.yml">FC<br>+new_self_critical</a></td>
<td align="center">1.053</td>
<td align="center">0.1857</td>
<td align="center"><a href="https://drive.google.com/open?id=1OsB_jLDorJnzKz6xsOfk1n493P3hwOP0">model&metrics</a></td>
<td align="center">--caption_model newfc</td>
</tr>
</tbody></table>

# Trained with Bottomup feature (10-100 features per image, not 36 features per image):

Collection: [link](https://drive.google.com/open?id=1-RNak8qLUR5LqfItY6OenbRl8sdwODng)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Note</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/a2i2.yml">Att2in</a></td>
<td align="center">1.089</td>
<td align="center">0.1982</td>
<td align="center"><a href="https://drive.google.com/open?id=1jO9bSocC93n1vBZmZVaASWc_jJ1VKZUq">model&metrics</a></td>
<td align="center">My replication</td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_sc.yml">Att2in<br>+self_critical</a></td>
<td align="center">1.173</td>
<td align="center">0.2046</td>
<td align="center"><a href="https://drive.google.com/open?id=1aI7hYUmgRLksI1wvN9-895GMHz4yStHz">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_nsc.yml">Att2in<br>+new_self_critical</a></td>
<td align="center">1.195</td>
<td align="center">0.2066</td>
<td align="center"><a href="https://drive.google.com/open?id=1BkxLPL4SuQ_qFa-4fN96u23iTFWw-iXX">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown.yml">UpDown</a></td>
<td align="center">1.099</td>
<td align="center">0.1999</td>
<td align="center"><a href="https://drive.google.com/open?id=14w8YXrjxSAi5D4Adx8jgfg4geQ8XS8wH">model&metrics</a></td>
<td align="center">My replication</td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown_sc.yml">UpDown<br>+self_critical</a></td>
<td align="center">1.227</td>
<td align="center">0.2145</td>
<td align="center"><a href="https://drive.google.com/open?id=1QdCigVWdDKTbUe3_HQFEGkAsv9XIkKkE">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown_nsc.yml">UpDown<br>+new_self_critical</a></td>
<td align="center">1.239</td>
<td align="center">0.2154</td>
<td align="center"><a href="https://drive.google.com/open?id=1cgoywxAdzHtIF2C6zNnIA7G2wjol_ybf">model&metrics</a></td>
<td align="center"></td>
</tr>
 <tr><td align="left"><a href="configs/updown/ud_long_nsc.yml">UpDown<br>+Schedule long<br>+new_self_critical</a></td>
<td align="center">1.280</td>
<td align="center">0.2200</td>
<td align="center"><a href="https://drive.google.com/open?id=1bCDmf4JCM79f5Lqp6MAn1ap4b3NJ5Gis">model&metrics</a></td>
<td align="center">Best of 5 models<br>schedule proposed by yangxuntu</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer.yml">Transformer</a></td>
<td align="center">1.1259</td>
<td align="center">0.2063</td>
<td align="center"><a href="https://drive.google.com/open?id=10Q5GJ2jZFCexD71rY9gg886Aasuaup8O">model&metrics</a></td>
<td align="center"></td>
</tr>
<tr><td align="left"><a href="configs/transformer/transformer_step.yml">Transformer(warmup+step decay)</a></td>
<td align="center">1.1496</td>
<td align="center">0.2093</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1Qog9yvpGWdHanFXFITjyrWXMzre3ek3e?usp=sharing">model&metrics</a></td>
<td align="center">Although this schedule is better, the final self critical results are similar.</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer_scl.yml">Transformer<br>+self_critical</a></td>
<td align="center">1.277</td>
<td align="center">0.2249</td>
<td align="center"><a href="https://drive.google.com/open?id=12iKJJSIGrzFth_dJXqcXy-_IjAU0I3DC">model&metrics</a></td>
<td align="center">This could be higher in my opinion. I chose the checkpoint with the highest CIDEr on val set, so it's possible some other checkpoint may perform better. Just let you know.</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer_nscl.yml">Transformer<br>+new_self_critical</a></td>
<td align="center"><b>1.303</b></td>
<td align="center">0.2289</td>
<td align="center"><a href="https://drive.google.com/open?id=1sJDqetTVOnei6Prgvl_4vkvrYlKlc-ka">model&metrics</a></td>
<td align="center"></td>
</tr>
</tbody></table>


# Trained with vilbert-12-in-1 feature:

Collection: [link](https://drive.google.com/drive/folders/1QdqRGUoPoQChOq65ecIaSl1yXOosSQm3?usp=sharing)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Note</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/transformer.yml">Transformer</a></td>
<td align="center">1.158</td>
<td align="center">0.2114</td>
<td align="center"><a href="https://drive.google.com/drive/folders/18tFqIgC1dc8KrRt71mY_CXaOM5Sr6b3k?usp=sharing">model&metrics</a></td>
<td align="center">The config needs to be changed to use the vilbert feature.</td>
</tr>
 <!-- <tr><td align="left"><a href="configs/transformer_nsc.yml">Transformer<br>+new_self_critical</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"><a href="">model&metrics</a></td>
<td align="center"></td>
</tr> -->
</tbody></table>
