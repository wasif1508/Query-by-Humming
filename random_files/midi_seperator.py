from glob import glob
import pretty_midi


def midi_sep():
    data_dird = '../mid_temp/'
    audio_files = glob(data_dird + '/*.mid')

    print(audio_files[2])
    midi_data = pretty_midi.PrettyMIDI(audio_files[2])


    note_no = []
    noteon = []
    noteoff = []
   
    for j in range(len(midi_data.instruments)):
        midi_data.instruments[j].remove_invalid_notes()
        if not midi_data.instruments[j].is_drum:
            if midi_data.instruments[j].program in range(0,7):
                for note in midi_data.instruments[j].notes:
                    note_no.append(note.pitch)
                    noteon.append(note.start)
                    noteoff.append(note.end)

    return note_no

print(midi_sep())