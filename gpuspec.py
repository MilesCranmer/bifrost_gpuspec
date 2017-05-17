import bifrost

import bifrost.pipeline as bfp
import bifrost.blocks as blocks
import bifrost.views as views
import bifrost.guppi_raw as guppi_raw

def get_with_default(obj, key, default=None):
	return obj[key] if key in obj else default

def mjd2unix(mjd):
	return (mjd - 40587) * 86400

class GuppiRawSourceBlock(bfp.SourceBlock):
    def __init__(self, sourcenames, gulp_nframe=1, *args, **kwargs):
        super(GuppiRawSourceBlock, self).__init__(sourcenames,
                              gulp_nframe=gulp_nframe,
                              *args, **kwargs)
    def create_reader(self, sourcename):
        return open(sourcename, 'rb')
    def on_sequence(self, reader, sourcename):
        previous_pos = reader.tell()
        ihdr = guppi_raw.read_header(reader)
        header_size = reader.tell() - previous_pos
        self.header_buf = bytearray(header_size)
        nbit      = ihdr['NBITS']
        assert(nbit in set([4,8,16,32,64]))
        nchan     = ihdr['OBSNCHAN']
        bw_MHz    = ihdr['OBSBW']
        cfreq_MHz = ihdr['OBSFREQ']
        df_MHz = bw_MHz / nchan
        f0_MHz = cfreq_MHz - 0.5*(nchan-1)*df_MHz
        # Note: This will be negative if OBSBW is negative, which is correct
        dt_s   = 1. / df_MHz / 1e6
        # Derive the timestamp of this block
        byte_offset   = ihdr['PKTIDX'] * ihdr['PKTSIZE']
        frame_nbyte   = ihdr['BLOCSIZE'] / ihdr['NTIME']
        bytes_per_sec = frame_nbyte / dt_s
        offset_secs   = byte_offset / bytes_per_sec
        tstart_mjd    = ihdr['STT_IMJD'] + (ihdr['STT_SMJD'] + offset_secs) / 86400.
        tstart_unix   = mjd2unix(tstart_mjd)
        ohdr = {
            '_tensor': {
                'dtype':  'ci' + str(nbit),
                'shape':  [-1, nchan, ihdr['NTIME'], ihdr['NPOL']],
                # Note: 'time' (aka block) is the frame axis
                'labels': ['time', 'freq', 'fine_time', 'pol'],
                'scales': [(tstart_unix, abs(dt_s)*ihdr['NTIME']),
                       (f0_MHz, df_MHz),
                       (0, dt_s),
                       None],
                'units':  ['s', 'MHz', 's', None]
            },
            'az_start':      get_with_default(ihdr, 'AZ'),        # Decimal degrees
            'za_start':      get_with_default(ihdr, 'ZA'),        # Decimal degrees
            'raj':       get_with_default(ihdr, 'RA')*(24./360.), # Decimal hours
            'dej':       get_with_default(ihdr, 'DEC'),       # Decimal degrees
            'source_name':   get_with_default(ihdr, 'SRC_NAME'),
            'refdm':     get_with_default(ihdr, 'CHAN_DM'),
            'refdm_units':   'pc cm^-3',
            'telescope':     get_with_default(ihdr, 'TELESCOP'),
            'machine':       get_with_default(ihdr, 'BACKEND'),
            'rawdatafile':   sourcename,
            'coord_frame':   'topocentric',
        }
        # Note: This gives 32 bits to the fractional part of a second,
        #     corresponding to ~0.233ns resolution. The whole part
        #     gets at least 31 bits, which will overflow in 2038.
        time_tag  = int(round(tstart_unix * 2**32))
        ohdr['time_tag'] = time_tag
        self.already_read_header = True
        
        ohdr['name'] = sourcename
        return [ohdr]
    def on_data(self, reader, ospans):
        if not self.already_read_header:
            # Skip over header
            #ihdr = guppi_raw.read_header(reader)
            nbyte = reader.readinto(self.header_buf)
            if nbyte == 0:
                return [0] # EOF
            elif nbyte < len(self.header_buf):
                raise IOError("Block header is truncated")
            self.already_read_header = False
            ospan = ospans[0]
            odata = ospan.data
            nbyte = reader.readinto(odata)
            if nbyte % ospan.frame_nbyte:
                #return [0]
                raise IOError("Block data is truncated")
            nframe = nbyte // ospan.frame_nbyte
            #print "nframe:", nframe
            #print "nbyte:", nbyte
            return [nframe]
        else:
            #only read one time step!
            reader.close()
            return [0]

def new_read_guppi_raw(filenames, *args, **kwargs):
    return GuppiRawSourceBlock(filenames, *args, **kwargs)

with bfp.Pipeline() as pipeline:
    raw_guppi = new_read_guppi_raw(['blc1_guppi_57388_HIP113357_0010.0000.raw'])
    g_guppi = blocks.copy(raw_guppi, space='cuda')
    ffted = blocks.fft(g_guppi, axes='fine_time', axis_labels='fft_freq')
    blocks.print_header(ffted)
    pipeline.run()
