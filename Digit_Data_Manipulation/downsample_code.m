
uiimport("C:\Users\camer\Audio_Data48\listnames.txt");
filenames = CUserscamerAudio_Data48010_01_0wav; %30000x1

for n = 1:30000 %from first to last row 
    xstr = filenames(n,1); %one element at a time to access each name
    [x, Fs] = audioread(xstr); %read in file name
    z = [x; zeros(8000,1)]; %column vector added 400 zeros vertically to end of x
    y = z'; %row vector
    yy = lowpass(y,0.29,Steepness=0.95);
    yd = downsample(yy,3); %downsample to 16000
    charname = char(xstr); %converts from string to char array
    newname = charname(32:end); %this gives me the file name
    i = charname(32); %this gives me the digit specifier
    audiowrite("C:\Users\camer\Audio_Data16\"+string(i)+"\"+string(newname),yd, 16000);
    
end

[H,w] = dtft(yy, 1000000);
plot(w/pi,20*log10(abs(H)));

