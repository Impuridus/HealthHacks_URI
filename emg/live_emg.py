"""Print live wearable EMG. Install: python -m pip install pylsl

Run: python live_emg.py
Use --every 1 to print every sample; default prints one in 80 to keep up.
"""
import argparse
from pylsl import StreamInlet, resolve_byprop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='Wave Band 2 C0403688 - emg')
    parser.add_argument('--every', type=int, default=80,
                        help='Print one in N received samples (default: 80)')
    args = parser.parse_args()
    if args.every < 1:
        parser.error('--every must be at least 1')
    print(f'Searching for {args.name!r}...', flush=True)
    streams = resolve_byprop('name', args.name, timeout=8)
    if not streams:
        raise SystemExit('Stream not found. Start the wearable LSL publisher and retry.')
    inlet = StreamInlet(streams[0], max_buflen=10)
    try:
        info = inlet.info(timeout=5)
        print(f'Connected: {info.name()} | {info.channel_count()} channels | '
              f'{info.nominal_srate():g} Hz advertised', flush=True)
        print(f'Values are in unspecified stream units. Printing 1 in {args.every} '
              'received samples. Ctrl+C stops.\n', flush=True)
        print('LSL timestamp | ' + ' | '.join(
            f'Channel {i + 1}' for i in range(info.channel_count())), flush=True)
        count = 0
        while True:
            samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=4096)
            if not timestamps:
                print('Waiting for samples...', flush=True)
                continue
            for timestamp, sample in zip(timestamps, samples):
                if count % args.every == 0:
                    print(f'{timestamp:.6f} | ' + ' | '.join(
                        f'{value:12.4f}' for value in sample), flush=True)
                count += 1
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
    finally:
        inlet.close_stream()


if __name__ == '__main__':
    main()
