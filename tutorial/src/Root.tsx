import React from 'react';
import {Composition, Still} from 'remotion';
import rawData from './data.json';
import {FPS} from './theme';
import {Data} from './types';
import {Tutorial, totalFrames} from './Tutorial';
import {Hero, Preview, previewFrames} from './Preview';

// All figures come from data.json (written by make_data.py). Pass --props to render with another file.
const data = rawData as unknown as Data;

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="Tutorial"
      component={Tutorial}
      width={1920}
      height={1080}
      fps={FPS}
      durationInFrames={totalFrames(data)}
      defaultProps={{data}}
      calculateMetadata={({props}) => ({durationInFrames: totalFrames(props.data)})}
    />
    <Composition
      id="Preview"
      component={Preview}
      width={960}
      height={540}
      fps={FPS}
      durationInFrames={previewFrames}
      defaultProps={{data}}
    />
    <Still id="Hero" component={Hero} width={1600} height={800} defaultProps={{data}} />
  </>
);
