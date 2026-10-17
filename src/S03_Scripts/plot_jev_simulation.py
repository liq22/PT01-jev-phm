"""Plot saved simulation results only: no training, no signal regeneration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

NOTE='MODEL-BASED SIMULATION | NOT MEASURED MACHINE DATA'

def main(run, out):
    run, out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams['svg.fonttype']='none'
    cfg=json.loads((run/'config.json').read_text())
    metrics=pd.read_csv(run/'metrics.csv')
    def save(fig,name,note=''):
        fig.subplots_adjust(bottom=.22,top=.82)
        fig.text(.12,.95,NOTE,fontsize=10)
        fig.text(.12,.04,note,fontsize=9,wrap=True)
        for ext in ('svg','png'):
            fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    signal=np.load(run/'signal_examples.npz');waves=signal['wave'];fs=int(signal['fs'])
    names=['Healthy','Inner-ring fault','Outer-ring fault']
    fig,ax=plt.subplots(figsize=(10,5))
    take=np.arange(int(.12*fs))
    for i,w in enumerate(waves):
        ax.plot(take/fs*1000,w[take]/w.std()*.19+i,lw=.8,label=names[i])
    ax.set(xlabel='Time (ms)',ylabel='Display-normalized traces, vertically offset',title='Simulated bearing vibration at the primary target condition',yticks=[0,1,2],yticklabels=names)
    save(fig,'01_waveforms','First predeclared virtual unit of each class, replicate 101, reference SNR = 0 dB.\nNormalization is for display only; training uses the declared common representation.')
    fig,ax=plt.subplots(figsize=(10,5))
    freq=np.fft.rfftfreq(waves.shape[1],1/fs)
    for w,name in zip(waves,names):
        power=np.abs(np.fft.rfft((w-w.mean())*np.hanning(len(w))))**2
        ax.plot(freq/1000,10*np.log10(np.maximum(power/power.max(),1e-10)),lw=.8,label=name)
    ax.set(xlabel='Frequency (kHz)',ylabel='Power relative to each trace maximum (dB)',title='Broadband vibration spectra',xlim=(0,8),ylim=(-80,3));ax.legend(loc='lower center',ncol=3)
    save(fig,'02_spectra','Same saved waveform examples; the resonant carrier is not the fault repetition frequency.\nSignals are from the reduced impulse-response model, not calibrated accelerometer measurements.')
    fig,ax=plt.subplots(figsize=(10,5))
    for w,fr,name in zip(waves,signal['shaft_hz'],names):
        filtered=sosfiltfilt(butter(4,cfg['bandpass_hz'],fs=fs,btype='bandpass',output='sos'),w)
        env=np.abs(hilbert(filtered))**2;env-=env.mean()
        power=np.abs(np.fft.rfft(env*np.hanning(len(env))))**2
        domain=(freq/fr>.5)&(freq/fr<12)
        ax.plot(freq[domain]/fr,10*np.log10(np.maximum(power[domain]/power[domain].max(),1e-9)),lw=1,label=name)
    for order,label in ((3.2,'BPFO: 3.2'),(4.8,'BPFI: 4.8')):
        ax.axvline(order,ls='--',lw=1);ax.text(order+.08,-2,label,rotation=90,va='top',fontsize=9)
    ax.set(xlabel='Order (frequency / observed shaft frequency)',ylabel='Squared-envelope spectrum (dB)',title='Fault repetition orders emerge after envelope demodulation',xlim=(.5,12),ylim=(-65,3));ax.legend(loc='lower right')
    save(fig,'03_envelope_orders','Order lines follow the declared 8-element, diameter-ratio 0.2 bearing geometry.\nThe classifier sees a full fixed order-bin grid, not per-record fault identity or generated impulse times.')
    fig,ax=plt.subplots(figsize=(10,4.6))
    labels={'joint_event':'Fine-only = event = Libra = soft anchor', 'conditional_freeze':'Fixed conditionals + coarse residual','rf_fine':'Tuned random forest (fine source only)'}
    prim=metrics[metrics.condition=='snr_0']
    for i,(method,name) in enumerate(labels.items()):
        vals=prim[prim.method==method].nll.to_numpy()
        ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.08,.08,len(vals)),s=40,label='Independent simulation cohorts' if i==0 else None)
        ax.scatter([vals.mean()],[i],marker='D',s=65,label='Mean across cohorts' if i==0 else None)
    ax.set(xscale='log',xlabel='Target negative log-likelihood (nats, log scale)',yticks=range(3),yticklabels=list(labels.values()),title='Primary comparison: the selected soft constraint is inactive')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.19),ncol=2,fontsize=9)
    save(fig,'04_primary_nll','Three independently generated cohorts; 90 virtual test units per cohort, four windows per unit.\nAll three soft-anchor selections use lambda = 0 and update step 0. Coincidence is not evidence of universal equivalence.')
    searches=json.loads((run/'searches.json').read_text())
    fig,ax=plt.subplots(figsize=(10,5))
    for penalty in (0.,.3,3.):
        histories=[]
        for seed in cfg['seeds']:
            rec=next(r for r in searches[str(seed)]['conditional_anchor']['trials'] if r['ridge']==.001 and r['anchor']==penalty)
            histories.append([r['within_fault_logodds_drift'] for r in rec['history']])
        steps=[r['step'] for r in rec['history']]
        ax.plot(steps,np.mean(histories,axis=0),marker='o',markevery=4,label=f'Conditional KL weight = {penalty:g}')
    ax.set(xlabel='Executed joint-source update steps',ylabel='Mean absolute inner/outer log-odds change',title='Executed candidate trials: drift control is not selected-model improvement')
    ax.legend()
    save(fig,'05_mechanism_trials','Fixed ridge = 0.001; same initialization and train/tune inputs. Curves average all coarse-tuning observations.\nThese are saved trial diagnostics, not test-selected checkpoints. Final selected models return to step 0.')
    fig,ax=plt.subplots(figsize=(10,5))
    for method,name in labels.items():
        means=[];sds=[]
        for snr in cfg['target_snr_db']:
            vals=metrics[(metrics.condition==f'snr_{snr}')&(metrics.method==method)].accuracy*100
            means.append(vals.mean());sds.append(vals.std())
        ax.errorbar(cfg['target_snr_db'],means,yerr=sds,marker='o',capsize=4,label=name)
    ax.set(xlabel='Reference SNR (dB; class-independent nominal response)',ylabel='Classification accuracy (%)',title='Noise sensitivity with source-selected models held fixed',xticks=cfg['target_snr_db'],ylim=(25,104));ax.legend(loc='lower right',fontsize=9)
    save(fig,'06_noise_boundary','Error bars are SD across three simulated cohorts, not confidence intervals for industrial machines.\nNoise conditions are paired perturbations of the same virtual units; they are not extra independent devices.')
    print(f'Saved six independent figures (SVG + PNG) to {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.run,a.out)
