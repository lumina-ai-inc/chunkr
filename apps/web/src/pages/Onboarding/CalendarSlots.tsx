import { Text } from "@radix-ui/themes";
import { useState, useLayoutEffect, useMemo } from "react";

interface TimeSlot {
  start: string;
}

interface CalendarData {
  [date: string]: TimeSlot[];
}

interface CalendarSlotsProps {
  calendarData: CalendarData;
  calendarError: string | null;
  selectedDate?: string;
  selectedTime?: string;
  onSlotSelect?: (date: string, time: string) => void;
  showScrollable?: boolean;
}

const formatDateDisplay = (dateStr: string) => {
  const parts = dateStr.split("-");
  let date: Date;

  if (parts.length === 3) {
    const year = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10) - 1;
    const day = parseInt(parts[2], 10);
    date = new Date(year, month, day);
  } else {
    date = new Date(dateStr);
  }
  const options: Intl.DateTimeFormatOptions = {
    month: "long",
    day: "numeric",
  };
  const formattedDate = date.toLocaleDateString("en-US", options);
  return formattedDate;
};

const formatTimeDisplay = (isoString: string | Date) => {
  const date = isoString instanceof Date ? isoString : new Date(isoString);
  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
};

const getUserTimezone = () => {
  const date = new Date();
  const timeString = date.toLocaleTimeString("en-US", {
    timeZoneName: "short",
  });
  return timeString.split(" ")[2] || "Local Time";
};

const convertToLocalTime = (isoString: string) => {
  return new Date(isoString);
};

const findFirstAvailableSlotForTomorrow = (
  calendarData: CalendarData
): { date: string; time: string } | null => {
  const availableDates = Object.keys(calendarData).sort();

  for (const date of availableDates) {
    const slots = calendarData[date];
    if (slots && slots.length > 0) {
      return {
        date,
        time: slots[0].start,
      };
    }
  }
  return null;
};

export default function CalendarSlots({
  calendarData,
  calendarError,
  selectedDate,
  selectedTime,
  onSlotSelect,
  showScrollable = false,
}: CalendarSlotsProps) {
  const [initialSelectionDone, setInitialSelectionDone] = useState(false);
  const userTimezone = useMemo(() => getUserTimezone(), []);

  useLayoutEffect(() => {
    if (
      !selectedDate &&
      !selectedTime &&
      onSlotSelect &&
      !initialSelectionDone &&
      Object.keys(calendarData).length > 0
    ) {
      const firstSlot = findFirstAvailableSlotForTomorrow(calendarData);
      if (firstSlot) {
        onSlotSelect(firstSlot.date, firstSlot.time);
      }
      setInitialSelectionDone(true);
    }
  }, [
    calendarData,
    selectedDate,
    selectedTime,
    onSlotSelect,
    initialSelectionDone,
  ]);

  const handleSlotClick = (date: string, timeSlot: TimeSlot) => {
    if (onSlotSelect) {
      onSlotSelect(date, timeSlot.start);
    }
  };

  const handleChangeTimeSlot = () => {
    if (onSlotSelect) {
      onSlotSelect("", "");
    }
  };

  const calendarEntries = useMemo(() => {
    return Object.entries(calendarData).map(([date, slots]) => ({
      date,
      slots,
      formattedDate: formatDateDisplay(date),
    }));
  }, [calendarData]);

  if (calendarError) {
    return (
      <div className="bg-red-500/20 border border-red-500 border-l-4 pl-3 py-2 rounded-md">
        <Text size="2" className="text-red-600">
          {calendarError}
        </Text>
      </div>
    );
  }

  if (selectedDate && selectedTime) {
    const formattedDate = formatDateDisplay(selectedDate);
    const formattedTime = formatTimeDisplay(convertToLocalTime(selectedTime));

    return (
      <div className="text-left flex flex-row gap-2 items-center">
        <Text size="3" weight="medium" className="text-white block mb-2">
          Your 15-minute call is scheduled for {formattedDate} at{" "}
          {formattedTime}. You will get a calendar invite.
          <button
            onClick={handleChangeTimeSlot}
            className="underline underline-offset-4 ml-2 text-white/70"
          >
            Change time slot
          </button>
        </Text>
      </div>
    );
  }

  return (
    <div
      className={`w-full ${
        showScrollable
          ? ""
          : "max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-sm"
      }`}
    >
      <div className="sticky top-0 z-20 bg-black py-2">
        <Text size="3" weight="medium" className="text-white/70">
          Times are in your local timezone ({userTimezone})
        </Text>
      </div>

      <div className="space-y-6">
        {calendarEntries.map(({ date, slots, formattedDate }) => (
          <div key={date}>
            <div className="sticky top-10 z-10 bg-black py-2">
              <Text size="3" weight="medium" className="text-white ">
                {formattedDate}
              </Text>
            </div>

            <div className="grid grid-cols-4 gap-3 px-4">
              {slots.map((slot, index) => {
                const localTime = convertToLocalTime(slot.start);
                const timeDisplay = formatTimeDisplay(localTime);
                const isSelected =
                  selectedDate === date && selectedTime === slot.start;

                return (
                  <button
                    key={index}
                    onClick={() => handleSlotClick(date, slot)}
                    className={`
                      !w-full !h-fit !px-4 !py-2 !rounded-md !transition-all !duration-200 !text-sm !font-medium
                      ${isSelected ? "button-active" : "button-resting"}
                    `}
                  >
                    {timeDisplay}
                  </button>
                );
              })}
            </div>

            {slots.length === 0 && (
              <div className="text-center py-8">
                <Text size="3" weight="medium" className="text-gray-500">
                  No available slots for this date
                </Text>
              </div>
            )}
          </div>
        ))}
      </div>
      {Object.keys(calendarData).length === 0 && (
        <div className="text-center py-8">
          <Text size="3" weight="medium" className="text-gray-500">
            No available dates found
          </Text>
        </div>
      )}
    </div>
  );
}
